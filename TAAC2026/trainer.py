"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import math
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class DenseParameterEMA:
    """Exponential moving average over trainable dense parameters only.

    The sparse embedding tables in this CTR-style model are intentionally
    excluded: cloning EMA shadows for all sparse embeddings can cost a large
    amount of GPU memory and often over-smooths high-cardinality IDs.  The EMA
    is therefore applied to the dense backbone / projection / fusion / head
    parameters, then swapped in only for validation and checkpoint saving.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        start_steps: int = 200,
        update_every: int = 1,
        sparse_param_ids: Optional[set] = None,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        if start_steps < 0:
            raise ValueError(f"EMA start_steps must be >= 0, got {start_steps}")
        if update_every <= 0:
            raise ValueError(f"EMA update_every must be > 0, got {update_every}")

        self.decay = float(decay)
        self.start_steps = int(start_steps)
        self.update_every = int(update_every)
        self.sparse_param_ids = sparse_param_ids or set()
        self.total_steps = 0
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        self.param_names = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and p.is_floating_point() and id(p) not in self.sparse_param_ids
        ]

        if not self.param_names:
            logging.warning("EMA requested but no dense trainable parameters were found; disabling EMA updates")

    @property
    def is_ready(self) -> bool:
        """Whether EMA shadows have been initialized at least once."""
        return self.num_updates > 0 and bool(self.shadow)

    def _named_ema_params(self, model: nn.Module):
        """Yield ``(name, param)`` pairs that participate in EMA."""
        wanted = set(self.param_names)
        for name, p in model.named_parameters():
            if name in wanted:
                yield name, p

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadows after an optimizer step."""
        if not self.param_names:
            return
        self.total_steps += 1
        if self.total_steps < self.start_steps:
            return
        if (self.total_steps - self.start_steps) % self.update_every != 0:
            return

        if not self.shadow:
            # Initialize from a reasonably trained model rather than from the
            # random/early warmup weights.
            for name, p in self._named_ema_params(model):
                self.shadow[name] = p.detach().clone()
            self.num_updates = 1
            logging.info(
                f"EMA initialized at step {self.total_steps}: "
                f"dense_params={len(self.shadow)}, decay={self.decay}, "
                f"update_every={self.update_every}"
            )
            return

        one_minus_decay = 1.0 - self.decay
        for name, p in self._named_ema_params(model):
            shadow = self.shadow[name]
            if shadow.device != p.device:
                shadow = shadow.to(device=p.device)
                self.shadow[name] = shadow
            if shadow.dtype != p.dtype:
                # Model weights are expected to be fp32; keep this guard so the
                # EMA remains safe under future dtype changes.
                shadow = shadow.to(dtype=p.dtype)
                self.shadow[name] = shadow
            shadow.mul_(self.decay).add_(p.detach(), alpha=one_minus_decay)
        self.num_updates += 1

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Backup current dense parameters before swapping in EMA weights."""
        self.backup = {
            name: p.detach().clone()
            for name, p in self._named_ema_params(model)
            if name in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA shadows into the live model."""
        if not self.is_ready:
            return
        for name, p in self._named_ema_params(model):
            shadow = self.shadow.get(name)
            if shadow is not None:
                p.copy_(shadow.to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore live dense parameters after EMA validation/checkpointing."""
        if not self.backup:
            return
        for name, p in self._named_ema_params(model):
            backup = self.backup.get(name)
            if backup is not None:
                p.copy_(backup.to(device=p.device, dtype=p.dtype))
        self.backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        """Temporarily evaluate/save the model with EMA dense parameters."""
        if not self.is_ready:
            yield False
            return
        self.store(model)
        self.copy_to(model)
        try:
            yield True
        finally:
            self.restore(model)


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        dense_weight_decay: float = 0.01,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_amp: bool = False,
        warmup_steps: int = 0,
        use_cosine_decay: bool = False,
        cosine_total_steps: int = 0,
        cosine_min_lr_ratio: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_start_steps: int = 200,
        ema_update_every: int = 1,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr}, wd={dense_weight_decay})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98), weight_decay=dense_weight_decay
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=dense_weight_decay
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.label_smoothing: float = float(label_smoothing)
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.dense_weight_decay: float = dense_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.use_amp: bool = use_amp and self.device.startswith('cuda')
        self.scaler = torch.amp.GradScaler(
            'cuda', enabled=False
        )

        # EMA over dense parameters only.  This intentionally excludes the large
        # sparse embedding tables so the feature is safe under the 19.2GB GPU
        # memory budget and does not over-smooth high-cardinality ID embeddings.
        self.use_ema: bool = bool(use_ema)
        self.ema: Optional[DenseParameterEMA] = None
        if self.use_ema:
            sparse_param_ids = set()
            if hasattr(model, 'get_sparse_params'):
                sparse_param_ids = {id(p) for p in model.get_sparse_params()}
            self.ema = DenseParameterEMA(
                model=model,
                decay=ema_decay,
                start_steps=ema_start_steps,
                update_every=ema_update_every,
                sparse_param_ids=sparse_param_ids,
            )
            logging.info(
                f"EMA enabled: dense-only, decay={ema_decay}, "
                f"start_steps={ema_start_steps}, update_every={ema_update_every}, "
                f"tracked_params={len(self.ema.param_names)}"
            )
        else:
            logging.info("EMA disabled")

        # LR scheduler for dense AdamW. Warmup can run alone; cosine decay is
        # gated separately so the default schedule is warmup -> constant lr.
        # Active when warmup_steps > 0 or cosine is enabled; otherwise dense
        # AdamW runs at constant lr.
        # Applied to dense_optimizer only; sparse Adagrad keeps its constant lr.
        self.warmup_steps: int = warmup_steps
        self.use_cosine_decay: bool = use_cosine_decay
        self.cosine_total_steps: int = cosine_total_steps
        self.cosine_min_lr_ratio: float = cosine_min_lr_ratio
        self.lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR]
        if warmup_steps > 0 or use_cosine_decay:
            if use_cosine_decay and cosine_total_steps <= warmup_steps:
                raise ValueError(
                    f"cosine_total_steps ({cosine_total_steps}) must be > "
                    f"warmup_steps ({warmup_steps})"
                )
            W = warmup_steps
            T = cosine_total_steps
            r = cosine_min_lr_ratio

            def lr_lambda(step: int) -> float:
                if step < W:
                    return float(step) / float(max(1, W))
                if not use_cosine_decay:
                    return 1.0
                if step >= T:
                    return r
                progress = (step - W) / float(T - W)
                return r + (1.0 - r) * 0.5 * (1.0 + math.cos(math.pi * progress))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.dense_optimizer, lr_lambda=lr_lambda
            )
            if use_cosine_decay and warmup_steps > 0:
                logging.info(
                    f"LR scheduler: linear warmup {W} steps -> cosine decay to "
                    f"step {T} (min_lr_ratio={r}); applied to dense AdamW only"
                )
            elif use_cosine_decay:
                logging.info(
                    f"LR scheduler: cosine decay to step {T} "
                    f"(min_lr_ratio={r}); applied to dense AdamW only"
                )
            else:
                logging.info(
                    f"LR scheduler: linear warmup {W} steps -> constant lr; "
                    f"applied to dense AdamW only"
                )
        else:
            self.lr_scheduler = None
            logging.info("LR scheduler: disabled (warmup_steps=0); dense AdamW uses constant lr")

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"label_smoothing={self.label_smoothing}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        logging.info("AMP enabled=%s, dtype=bf16, grad_scaler=%s",
                     self.use_amp, self.scaler.is_enabled())

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, skip_model_file=True)

    def _evaluate_and_handle_checkpoint(
        self,
        epoch: int,
        total_step: int,
    ) -> Tuple[float, float]:
        """Validate and save best checkpoint with the same weights used for AUC.

        When EMA is ready, dense EMA weights are temporarily swapped into the
        live model for both validation and ``EarlyStopping`` checkpointing.
        The original training weights are restored immediately afterwards so
        subsequent optimizer steps continue from the non-EMA trajectory.
        """
        ema_ready = self.ema is not None and self.ema.is_ready
        if ema_ready:
            logging.info(
                f"Evaluating with EMA dense weights "
                f"(updates={self.ema.num_updates}, decay={self.ema.decay})"
            )
        else:
            logging.info("Evaluating with live model weights (EMA not ready yet)")

        if self.ema is None:
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self._handle_validation_result(total_step, val_auc, val_logloss)
            return val_auc, val_logloss

        with self.ema.average_parameters(self.model):
            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self._handle_validation_result(total_step, val_auc, val_logloss)
        return val_auc, val_logloss

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        # Smoothed train loss logging: window-mean every TRAIN_LOG_EVERY steps.
        # Per-step `Loss/train` is still written for fine-grained inspection.
        TRAIN_LOG_EVERY = 500
        window_sum = 0.0
        window_count = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                loss_sum += loss
                window_sum += loss
                window_count += 1

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)
                    self.writer.add_scalar(
                        'LR/dense',
                        self.dense_optimizer.param_groups[0]['lr'],
                        total_step,
                    )

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Smoothed train-loss point + console line every TRAIN_LOG_EVERY steps.
                if window_count >= TRAIN_LOG_EVERY:
                    window_avg = window_sum / window_count
                    if self.writer:
                        self.writer.add_scalar('Loss/train_smoothed', window_avg, total_step)
                    logging.info(
                        f"Step {total_step} | train loss avg(last {TRAIN_LOG_EVERY}): "
                        f"{window_avg:.6f}"
                    )
                    window_sum = 0.0
                    window_count = 0

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self._evaluate_and_handle_checkpoint(
                        epoch=epoch, total_step=total_step
                    )
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}")

            val_auc, val_logloss = self._evaluate_and_handle_checkpoint(
                epoch=epoch, total_step=total_step
            )
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        seq_delta_buckets: Dict[str, torch.Tensor] = {}
        seq_hour_buckets: Dict[str, torch.Tensor] = {}
        seq_dow_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_delta_buckets[domain] = device_batch.get(
                f'{domain}_delta_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_hour_buckets[domain] = device_batch.get(
                f'{domain}_hour_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
            seq_dow_buckets[domain] = device_batch.get(
                f'{domain}_dow_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            time_summary_feats=device_batch.get(
                'time_summary_feats',
                torch.zeros(
                    device_batch['user_int_feats'].shape[0],
                    len(seq_domains) * 8,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            seq_overflow_summary_feats=device_batch.get(
                'seq_overflow_summary_feats',
                torch.zeros(
                    device_batch['user_int_feats'].shape[0],
                    len(seq_domains) * 5,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
            seq_delta_buckets=seq_delta_buckets,
            seq_hour_buckets=seq_hour_buckets,
            seq_dow_buckets=seq_dow_buckets,
        )

    def _smooth_binary_labels(self, label: torch.Tensor) -> torch.Tensor:
        """Apply symmetric binary label smoothing for the training loss only.

        eps=0 keeps labels unchanged.  With eps>0, hard labels are moved
        toward 0.5: 0 -> eps/2 and 1 -> 1 - eps/2.  This is conservative
        for imbalanced CTR/PCVR data and avoids changing validation AUC/logloss.
        """
        if self.label_smoothing <= 0.0:
            return label
        return label * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()
        loss_label = self._smooth_binary_labels(label)

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        with torch.amp.autocast(
            'cuda', enabled=self.use_amp, dtype=torch.bfloat16
        ):
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(logits, loss_label, alpha=self.focal_alpha, gamma=self.focal_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, loss_label)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.unscale_(self.sparse_optimizer)
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.scaler.step(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.step(self.sparse_optimizer)
        self.scaler.update()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.ema is not None:
            self.ema.update(self.model)

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        with torch.amp.autocast(
            'cuda', enabled=self.use_amp, dtype=torch.bfloat16
        ):
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1).float()  # (B,)

        return logits, label
