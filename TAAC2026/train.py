"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import (
    FeatureSchema,
    USER_TIME_DOW_FID,
    USER_TIME_HOD_FID,
    get_pcvr_data,
    NUM_DELTA_BUCKETS,
    NUM_TIME_BUCKETS,
)
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--dense_weight_decay', type=float, default=0.01,
                        help='Weight decay for dense parameters (AdamW). '
                             'PyTorch AdamW default is 0.01; the historical baseline '
                             'implicitly used this value, so leaving it at 0.01 reproduces '
                             'baseline behavior exactly.')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable CUDA automatic mixed precision training')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile for the model')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--longer_gather_side', type=str, default='head',
                        choices=['head', 'tail'],
                        help='LongerEncoder Q gather side: '
                             'head = newest top_k tokens (correct semantic, default), '
                             'tail = oldest top_k tokens (legacy bugged behavior, kept for A/B). '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--per_domain_time_embeddings', action='store_true', default=False,
                        help='Use one recency time-bucket embedding table per sequence '
                             'domain while keeping the global bucket boundaries unchanged.')
    parser.add_argument('--domain_time_residual_embeddings', action='store_true', default=False,
                        help='Add zero-initialized per-domain residual time embeddings on '
                             'top of the shared recency time embedding.')
    parser.add_argument('--use_time_summary_features', action='store_true', default=False,
                        help='Add one NS token from per-domain sequence time summary features '
                             '(last/oldest/span/density/window counts).')
    parser.add_argument('--use_seq_overflow_summary_features', action='store_true', default=False,
                        help='Add one NS token from raw sequence length / truncation summary '
                             'features. This preserves information lost when long histories '
                             'are capped by --seq_max_lens.')
    parser.add_argument('--no_seq_overflow_summary_features',
                        dest='use_seq_overflow_summary_features',
                        action='store_false',
                        help='Disable sequence overflow summary features')
    parser.add_argument('--use_seq_periodic_time_features', action='store_true', default=False,
                        help='Concatenate hour-of-day and day-of-week embeddings to each '
                             'sequence token before the per-domain sequence projection.')
    parser.add_argument('--use_seq_hour_of_day_feature', action='store_true', default=False,
                        help='Concatenate hour-of-day embeddings to each sequence token.')
    parser.add_argument('--use_seq_day_of_week_feature', action='store_true', default=False,
                        help='Concatenate day-of-week embeddings to each sequence token.')
    parser.add_argument('--per_domain_seq_periodic_time_features', action='store_true', default=False,
                        help='Use per-domain hour-of-day and day-of-week embeddings for '
                             'sequence periodic time features.')
    parser.add_argument('--use_delta_buckets', action='store_true', default=False,
                        help='Enable per-domain delta-t bucket embedding (W2.7). '
                             'Models adjacent-token time gaps within sequences. '
                             'Bucket count is determined by dataset.NUM_DELTA_BUCKETS.')
    parser.add_argument('--no_delta_buckets', dest='use_delta_buckets', action='store_false',
                        help='Disable the delta-t bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Binary label smoothing strength for training loss only. '
                             '0.0 disables smoothing; eps maps labels as '
                             'y_smooth = y * (1 - eps) + 0.5 * eps.')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # LR scheduler for dense AdamW. Warmup is enabled by default; cosine decay
    # is opt-in so the default schedule is warmup -> constant lr.
    # Sparse Adagrad is unaffected.
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Linear warmup steps for dense AdamW '
                             '(0 = disable warmup).')
    parser.add_argument('--use_cosine_decay', action='store_true',
                        help='After warmup, enable cosine decay for dense AdamW. '
                             'Off by default; without this flag, lr stays at --lr '
                             'after warmup.')
    parser.add_argument('--cosine_total_epochs', type=float, default=8.0,
                        help='Cosine decay reaches its floor after this many epochs '
                             '(total_steps = epochs * len(train_loader)). '
                             'Effective only when --use_cosine_decay is set.')
    parser.add_argument('--cosine_min_lr_ratio', type=float, default=0.1,
                        help='Cosine floor expressed as a fraction of peak --lr '
                             '(0.1 means floor lr = 0.1 * --lr). '
                             'Effective only when --use_cosine_decay is set.')

    # EMA: enabled by default for the dense backbone/head only.  Sparse
    # embedding tables are deliberately excluded inside trainer.py to keep the
    # memory footprint low on the TAAC 19.2GB GPU.
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Enable dense-only EMA for validation and best checkpoint saving '
                             '(default: enabled).')
    parser.add_argument('--no_ema', dest='use_ema', action='store_false',
                        help='Disable EMA and validate/save live model weights.')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for dense parameters.')
    parser.add_argument('--ema_start_steps', type=int, default=200,
                        help='Start EMA updates after this many optimizer steps.')
    parser.add_argument('--ema_update_every', type=int, default=1,
                        help='Update EMA every N optimizer steps after --ema_start_steps.')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')
    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')
    parser.add_argument('--split_user_int_shared_fids', action='store_true',
                        help='In rankmixer mode, extract user-int fids '
                             '62/63/64/65/66/89/90/91 into one dedicated MLP '
                             'token and split the remaining user-int features '
                             'across user_ns_tokens-1 RankMixer tokens.')
    parser.add_argument('--use_dense_group_projector', action='store_true',
                        help='Project TAAC user dense features into two tokens: '
                             'fid 61/87 and fid 62/63/64/65/66/89/90/91. '
                             'When disabled, use the baseline single dense token.')
    parser.add_argument('--use_ui_attn_fusion', action='store_true', default=False,
                        help='Enable main fusion head: mean-pool user tokens as u_pool, '
                             'mean-pool item tokens as i_pool, use i_pool to attend over '
                             'all sequence tokens as i_attn, then classify from '
                             'concat(u_pool, i_pool, i_attn).')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')
    if args.per_domain_time_embeddings and args.domain_time_residual_embeddings:
        parser.error(
            "--per_domain_time_embeddings and --domain_time_residual_embeddings "
            "are mutually exclusive")

    return args


def main() -> None:
    args = parse_args()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_group_values = list(ns_groups_cfg['user_ns_groups'].values())
        user_group_values[-1] = user_group_values[-1] + [
            USER_TIME_DOW_FID,
            USER_TIME_HOD_FID,
        ]
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in user_group_values]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
        "item_dense_feature_specs": pcvr_dataset.item_dense_schema.entries,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "user_int_feature_ids": pcvr_dataset.user_int_schema.feature_ids,
        "item_int_feature_ids": pcvr_dataset.item_int_schema.feature_ids,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "seq_longer_gather_side": args.longer_gather_side,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "per_domain_time_embeddings": args.per_domain_time_embeddings,
        "domain_time_residual_embeddings": args.domain_time_residual_embeddings,
        "num_delta_buckets": NUM_DELTA_BUCKETS if args.use_delta_buckets else 0,
        "use_time_summary_features": args.use_time_summary_features,
        "use_seq_overflow_summary_features": args.use_seq_overflow_summary_features,
        "use_seq_hour_of_day_feature": args.use_seq_hour_of_day_feature,
        "use_seq_day_of_week_feature": args.use_seq_day_of_week_feature,
        "use_seq_periodic_time_features": args.use_seq_periodic_time_features,
        "per_domain_seq_periodic_time_features": args.per_domain_seq_periodic_time_features,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "split_user_int_shared_fids": args.split_user_int_shared_fids,
        "use_dense_group_projector": args.use_dense_group_projector,
        "use_ui_attn_fusion": args.use_ui_attn_fusion,
    }

    model = PCVRHyFormer(**model_args).to(args.device)
    if model.num_time_buckets > 0 and model.per_domain_time_embeddings:
        n_domains = len(model.seq_domains)
        logging.info(
            f"W2.7.1 per-domain recency time embeddings enabled: "
            f"{n_domains} x {NUM_TIME_BUCKETS} x {args.d_model}, "
            f"+{(n_domains - 1) * NUM_TIME_BUCKETS * args.d_model} params vs shared"
        )
    if model.num_time_buckets > 0 and model.domain_time_residual_embeddings:
        n_domains = len(model.seq_domains)
        logging.info(
            f"W2.7.2 domain residual recency time embeddings enabled: "
            f"shared + zero-init residual ({n_domains} x {NUM_TIME_BUCKETS} x "
            f"{args.d_model}), +{n_domains * NUM_TIME_BUCKETS * args.d_model} params"
        )
    if model.use_time_summary_features:
        logging.info(
            f"W2.8 time summary NS token enabled: "
            f"{len(model.seq_domains) * 8} dims -> 1 x {args.d_model}"
        )
    if model.use_seq_overflow_summary_features:
        logging.info(
            f"W3.0 seq overflow summary NS token enabled: "
            f"{len(model.seq_domains) * 5} dims -> 1 x {args.d_model}"
        )
    if model.use_seq_periodic_time_features:
        periodic_scope = (
            "per-domain"
            if model.per_domain_seq_periodic_time_features
            else "shared"
        )
        periodic_parts = []
        if model.use_seq_hour_of_day_feature:
            periodic_parts.append("hour-of-day")
        if model.use_seq_day_of_week_feature:
            periodic_parts.append("day-of-week")
        logging.info(
            f"W2.9 seq periodic time features enabled ({periodic_scope}): concat "
            f"{' + '.join(periodic_parts)} embeddings before seq projection"
        )
    if model.num_delta_buckets > 0:
        n_domains = len(model.seq_domains)
        logging.info(
            f"W2.7 delta_buckets enabled: NUM_DELTA_BUCKETS={NUM_DELTA_BUCKETS}, "
            f"per-domain ({n_domains} x {NUM_DELTA_BUCKETS} x {args.d_model}), "
            f"+{n_domains * NUM_DELTA_BUCKETS * args.d_model} params"
        )
    if args.use_compile:
        if hasattr(torch, 'compile'):
            try:
                logging.info(f"Compiling model.forward with torch.compile(mode={args.compile_mode})")
                model.forward = torch.compile(model.forward, mode=args.compile_mode)
            except Exception:
                logging.exception("torch.compile failed; falling back to eager model")
        else:
            logging.warning("torch.compile is not available in this PyTorch build")

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}, use_ui_attn_fusion={args.use_ui_attn_fusion}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    # Cosine schedule terminus, in steps. Computed here because trainer should
    # not have to peek at len(train_loader) just to convert epochs->steps.
    cosine_total_steps = int(args.cosine_total_epochs * len(train_loader))
    if args.warmup_steps > 0 or args.use_cosine_decay:
        logging.info(
            f"Warmup-LR config: warmup_steps={args.warmup_steps}, "
            f"use_cosine_decay={args.use_cosine_decay}, "
            f"cosine_total_steps={cosine_total_steps} "
            f"(={args.cosine_total_epochs} epochs * {len(train_loader)} steps/epoch), "
            f"cosine_min_lr_ratio={args.cosine_min_lr_ratio}"
        )

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        dense_weight_decay=args.dense_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
        use_amp=args.use_amp,
        warmup_steps=args.warmup_steps,
        use_cosine_decay=args.use_cosine_decay,
        cosine_total_steps=cosine_total_steps,
        cosine_min_lr_ratio=args.cosine_min_lr_ratio,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        ema_start_steps=args.ema_start_steps,
        ema_update_every=args.ema_update_every,
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
