#!/bin/bash
# ============================================================
# PCVRHyFormer + TIN  ——  全参数显式平铺版
#
# 所有超参直接以 --flag 形式声明,不再用 env-var 切换。需要消融某项
# 时直接编辑对应行(或追加 "$@" 透传到命令行覆盖)。
#
# 章节顺序:
#   1) 训练基础         seed / batch / epochs / patience / workers
#   2) 性能加速         AMP + torch.compile
#   3) 数据采样         train/valid ratio, eval cadence, seq_max_lens
#   4) 模型架构         d_model / queries / blocks / heads / dropout
#   5) 序列编码器       transformer / top_k / gather side
#   6) 时间特征         periodic + overflow summary
#   7) NS Tokenizer    rankmixer + 4 user / 2 item tokens
#   8) Embedding       skip_threshold / seq_id_threshold
#   9) 优化器 + LR      AdamW + 500 warmup
#  10) 稀疏参数         SparseAdam + reinit
#  11) EMA             decay 0.999, start@200
#  12) Loss            BCE
#  13) TIN             --use_tin 替换 DIN 主干
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --seed 42 \
    --batch_size 256 \
    --num_epochs 8 \
    --patience 5 \
    --num_workers 8 \
    --buffer_batches 20 \
    --use_amp \
    --use_compile \
    --compile_mode reduce-overhead \
    --train_ratio 1.0 \
    --valid_ratio 0.1 \
    --eval_every_n_steps 0 \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
    --d_model 68 \
    --emb_dim 64 \
    --num_queries 2 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --hidden_mult 4 \
    --dropout_rate 0.01 \
    --rank_mixer_mode full \
    --action_num 1 \
    --seq_encoder_type transformer \
    --seq_top_k 50 \
    --longer_gather_side head \
    --use_time_buckets \
    --domain_time_residual_embeddings \
    --use_delta_buckets \
    --use_seq_periodic_time_features \
    --per_domain_seq_periodic_time_features \
    --use_seq_overflow_summary_features \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --split_user_int_shared_fids \
    --use_dense_group_projector \
    --ns_groups_json "" \
    --emb_skip_threshold 1100000 \
    --seq_id_threshold 10000 \
    --lr 1e-4 \
    --dense_weight_decay 0.01 \
    --warmup_steps 500 \
    --sparse_lr 0.05 \
    --sparse_weight_decay 0.0 \
    --reinit_sparse_after_epoch 1 \
    --reinit_cardinality_threshold 0 \
    --use_ema \
    --ema_decay 0.999 \
    --ema_start_steps 200 \
    --ema_update_every 1 \
    --loss_type bce \
    --label_smoothing 0.01 \
    --focal_alpha 0.1 \
    --focal_gamma 2.0 \
    "$@"