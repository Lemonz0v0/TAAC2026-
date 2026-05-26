"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    time_summary_feats: torch.Tensor
    seq_overflow_summary_feats: torch.Tensor  # [B, num_domains * 5], raw-len/truncation summary
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_delta_buckets: dict  # {domain: tensor [B, L]} - W2.7 adjacent-token delta_t buckets
    seq_hour_buckets: dict  # {domain: tensor [B, L]} - hour-of-day ids, 0=padding
    seq_dow_buckets: dict  # {domain: tensor [B, L]} - day-of-week ids, 0=padding


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model

        global_info_dim = (num_ns + 1) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            global_info = torch.cat([ns_flat, seq_pooled], dim=-1)  # (B, (M+1)*D)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
        gather_side: str = 'head',
    ) -> None:
        super().__init__()
        if gather_side not in ('head', 'tail'):
            raise ValueError(f"gather_side must be 'head' or 'tail', got {gather_side!r}")
        self.top_k = top_k
        self.causal = causal
        self.gather_side = gather_side

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects top_k tokens from each sample.

        Sequences are stored in reverse-time order (pos 0 = most recent) with
        right-side padding. Two gather modes:

        - ``head`` (default, correct semantic): take positions ``[0, top_k)``
          = newest top_k tokens.
        - ``tail`` (legacy bugged behavior, kept for A/B): take positions
          ``[valid_len - top_k, valid_len)`` = oldest top_k tokens.

        Output layout is uniform: valid tokens at ``[0, n_valid)``, padding at
        ``[n_valid, top_k)``, where ``n_valid = min(valid_len, top_k)``. When
        ``valid_len <= top_k`` the two modes coincide (all valid tokens kept).

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample (number of non-padding tokens)
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)
        n_valid = torch.clamp(valid_len, max=self.top_k)  # (B,) actual valid count in output

        if self.gather_side == 'head':
            start_pos = torch.zeros_like(valid_len)
        else:  # 'tail'
            start_pos = torch.clamp(valid_len - self.top_k, min=0)

        offsets = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather tokens: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # Padding mask: positions >= n_valid are padding (valid-first layout)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices >= n_valid.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        position_indices = indices  # original positions (for Q-side RoPE)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False,
    gather_side: str = 'head',
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).
        gather_side: 'head' (newest top_k, correct) or 'tail' (oldest top_k,
            legacy bug behavior). Only used by longer.

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal, gather_side)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        longer_gather_side: str = 'head',
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal,
                gather_side=longer_gather_side,
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class SplitUserIntNSTokenizer(nn.Module):
    """User-int tokenizer with one selected-fid token plus RankMixer tokens."""

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        feature_ids: List[int],
        special_fids: List[int],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_regular_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.feature_ids = feature_ids
        self.special_fids = special_fids
        self.emb_dim = emb_dim
        self.num_regular_tokens = num_regular_tokens
        self.emb_skip_threshold = emb_skip_threshold

        fid_to_idx = {fid: idx for idx, fid in enumerate(feature_ids)}
        self.special_indices = [fid_to_idx[fid] for fid in special_fids]
        special_index_set = set(self.special_indices)
        self.regular_groups = [
            [fid_idx for fid_idx in group if fid_idx not in special_index_set]
            for group in groups
        ]
        self.regular_groups = [group for group in self.regular_groups if group]

        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        special_dim = len(self.special_indices) * emb_dim
        self.special_proj = nn.Sequential(
            nn.Linear(special_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        total_regular_fids = sum(len(g) for g in self.regular_groups)
        total_regular_dim = total_regular_fids * emb_dim
        self.chunk_dim = math.ceil(total_regular_dim / num_regular_tokens)
        self.padded_total_dim = self.chunk_dim * num_regular_tokens
        self._pad_size = self.padded_total_dim - total_regular_dim
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_regular_tokens)
        ])

        logging.info(
            f"SplitUserIntNSTokenizer: special_fids={special_fids}, "
            f"regular_fids={total_regular_fids}, special_dim={special_dim}, "
            f"regular_chunk_dim={self.chunk_dim}, regular_tokens={num_regular_tokens}, "
            f"pad={self._pad_size}"
        )

    def _embed_feature(self, int_feats: torch.Tensor, fid_idx: int) -> torch.Tensor:
        vs, offset, length = self.feature_specs[fid_idx]
        emb_real_idx = self._emb_index[fid_idx]
        if emb_real_idx == -1:
            return int_feats.new_zeros(int_feats.shape[0], self.emb_dim)

        emb_layer = self.embs[emb_real_idx]
        if length == 1:
            return emb_layer(int_feats[:, offset].long())

        vals = int_feats[:, offset:offset + length].long()
        emb_all = emb_layer(vals)
        mask = (vals != 0).float().unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        return (emb_all * mask).sum(dim=1) / count

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        special_embs = [
            self._embed_feature(int_feats, fid_idx)
            for fid_idx in self.special_indices
        ]
        special_token = F.silu(
            self.special_proj(torch.cat(special_embs, dim=-1))
        ).unsqueeze(1)

        regular_embs = []
        for group in self.regular_groups:
            for fid_idx in group:
                regular_embs.append(self._embed_feature(int_feats, fid_idx))

        cat_emb = torch.cat(regular_embs, dim=-1)
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))

        regular_tokens = []
        for chunk, proj in zip(cat_emb.split(self.chunk_dim, dim=-1), self.token_projs):
            regular_tokens.append(F.silu(proj(chunk)).unsqueeze(1))

        return torch.cat([special_token] + regular_tokens, dim=1)


class DenseGroupProjector(nn.Module):
    """Projects selected dense feature-id groups into NS tokens."""

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        d_model: int,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups

        fid_to_spec = {fid: (offset, length) for fid, offset, length in feature_specs}
        self._group_slices = []
        for group in groups:
            slices = [fid_to_spec[fid] for fid in group]
            self._group_slices.append(slices)

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sum(length for _, length in slices), d_model),
                nn.LayerNorm(d_model),
            )
            for slices in self._group_slices
        ])

        logging.info(
            "DenseGroupProjector: "
            + ", ".join(
                f"group={group}, dim={sum(length for _, length in slices)}"
                for group, slices in zip(groups, self._group_slices)
            )
        )

    @property
    def num_tokens(self) -> int:
        return len(self.groups)

    def forward(self, dense_feats: torch.Tensor) -> torch.Tensor:
        tokens = []
        for slices, proj in zip(self._group_slices, self.group_projs):
            group_feats = [
                dense_feats[:, offset:offset + length]
                for offset, length in slices
            ]
            cat_feats = torch.cat(group_feats, dim=-1)
            tokens.append(F.silu(proj(cat_feats)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        user_int_feature_ids: Optional[List[int]] = None,
        item_int_feature_ids: Optional[List[int]] = None,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        item_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        seq_longer_gather_side: str = 'head',
        action_num: int = 1,
        num_time_buckets: int = 65,
        per_domain_time_embeddings: bool = False,
        domain_time_residual_embeddings: bool = False,
        num_delta_buckets: int = 0,
        use_time_summary_features: bool = False,
        use_seq_overflow_summary_features: bool = False,
        use_seq_hour_of_day_feature: bool = False,
        use_seq_day_of_week_feature: bool = False,
        use_seq_periodic_time_features: bool = False,
        per_domain_seq_periodic_time_features: bool = False,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        split_user_int_shared_fids: bool = False,
        use_dense_group_projector: bool = False,
        use_ui_attn_fusion: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.per_domain_time_embeddings = per_domain_time_embeddings
        self.domain_time_residual_embeddings = domain_time_residual_embeddings
        self.num_delta_buckets = num_delta_buckets
        self.use_time_summary_features = use_time_summary_features
        self.use_seq_overflow_summary_features = use_seq_overflow_summary_features
        self.seq_overflow_summary_features_per_domain = 5
        self.use_seq_hour_of_day_feature = (
            use_seq_hour_of_day_feature
            or use_seq_periodic_time_features
            or per_domain_seq_periodic_time_features
        )
        self.use_seq_day_of_week_feature = (
            use_seq_day_of_week_feature
            or use_seq_periodic_time_features
            or per_domain_seq_periodic_time_features
        )
        self.use_seq_periodic_time_features = (
            self.use_seq_hour_of_day_feature
            or self.use_seq_day_of_week_feature
        )
        self.per_domain_seq_periodic_time_features = per_domain_seq_periodic_time_features
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.use_ui_attn_fusion = use_ui_attn_fusion

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            split_user_int_fids = [62, 63, 64, 65, 66, 89, 90, 91]
            user_int_fid_set = set(user_int_feature_ids or [])
            if (split_user_int_shared_fids and user_ns_tokens > 1
                    and set(split_user_int_fids).issubset(user_int_fid_set)):
                self.user_ns_tokenizer = SplitUserIntNSTokenizer(
                    feature_specs=user_int_feature_specs,
                    feature_ids=user_int_feature_ids or [],
                    special_fids=split_user_int_fids,
                    groups=user_ns_groups,
                    emb_dim=emb_dim,
                    d_model=d_model,
                    num_regular_tokens=user_ns_tokens - 1,
                    emb_skip_threshold=emb_skip_threshold,
                )
            else:
                if split_user_int_shared_fids:
                    logging.warning(
                        "split_user_int_shared_fids is enabled but the expected "
                        "TAAC user-int fids are unavailable or user_ns_tokens <= 1; "
                        "falling back to baseline RankMixer user-int tokenizer."
                    )
                self.user_ns_tokenizer = RankMixerNSTokenizer(
                    feature_specs=user_int_feature_specs,
                    groups=user_ns_groups,
                    emb_dim=emb_dim,
                    d_model=d_model,
                    num_ns_tokens=user_ns_tokens,
                    emb_skip_threshold=emb_skip_threshold,
                )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        self.user_dense_num_tokens = 0
        if self.has_user_dense:
            # Dense heat/profile regrouping based on train diagnostics:
            # fids 61/89/90 strongly track time/bin label-rate and global density,
            # so expose them as one dedicated NS token instead of mixing 61 with 87.
            # Keep two dense tokens total, preserving T=16 and d_model=64 compatibility.
            dense_heat_group = [61, 89, 90]
            dense_profile_group = [62, 63, 64, 65, 66, 87, 91]
            dense_fids = {
                fid for fid, _, _ in (user_dense_feature_specs or [])
            }
            if (use_dense_group_projector
                    and set(dense_heat_group + dense_profile_group).issubset(dense_fids)):
                self.user_dense_proj = DenseGroupProjector(
                    feature_specs=user_dense_feature_specs or [],
                    groups=[dense_heat_group, dense_profile_group],
                    d_model=d_model,
                )
            else:
                if use_dense_group_projector:
                    logging.warning(
                        "use_dense_group_projector is enabled but the expected "
                        "TAAC dense fids are unavailable; falling back to a "
                        "single dense NS token."
                    )
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )
            self.user_dense_num_tokens = (
                self.user_dense_proj.num_tokens
                if isinstance(self.user_dense_proj, DenseGroupProjector)
                else 1
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        self.item_dense_num_tokens = 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self.item_dense_num_tokens = 1

        # Total NS token count.  We keep explicit user/item spans so that the
        # final UI-attention fusion head can pool user tokens and item tokens
        # without depending on the concrete tokenizer implementation.
        self.num_user_ns = num_user_ns
        self.num_item_ns = num_item_ns
        self.user_token_start = 0
        self.user_token_end = num_user_ns + self.user_dense_num_tokens
        self.item_token_start = self.user_token_end
        self.item_token_end = self.item_token_start + num_item_ns + self.item_dense_num_tokens
        self.num_ns = (num_user_ns + self.user_dense_num_tokens
                       + num_item_ns + self.item_dense_num_tokens)
        if use_time_summary_features:
            self.num_ns += 1
        if use_seq_overflow_summary_features:
            self.num_ns += 1

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            periodic_feature_count = (
                int(self.use_seq_hour_of_day_feature)
                + int(self.use_seq_day_of_week_feature)
            )
            seq_proj_in_dim = (
                len(vs)
                + periodic_feature_count
            ) * emb_dim
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(seq_proj_in_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if self.use_seq_periodic_time_features:
            if per_domain_seq_periodic_time_features:
                if self.use_seq_hour_of_day_feature:
                    self.seq_hour_embeddings = nn.ModuleDict({
                        d: nn.Embedding(25, emb_dim, padding_idx=0)
                        for d in self.seq_domains
                    })
                if self.use_seq_day_of_week_feature:
                    self.seq_dow_embeddings = nn.ModuleDict({
                        d: nn.Embedding(8, emb_dim, padding_idx=0)
                        for d in self.seq_domains
                    })
            else:
                if self.use_seq_hour_of_day_feature:
                    self.seq_hour_embedding = nn.Embedding(25, emb_dim, padding_idx=0)
                if self.use_seq_day_of_week_feature:
                    self.seq_dow_embedding = nn.Embedding(8, emb_dim, padding_idx=0)

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            if per_domain_time_embeddings:
                self.time_embeddings = nn.ModuleDict({
                    d: nn.Embedding(num_time_buckets, d_model, padding_idx=0)
                    for d in self.seq_domains
                })
            else:
                self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)
                if domain_time_residual_embeddings:
                    self.time_residual_embeddings = nn.ModuleDict({
                        d: nn.Embedding(num_time_buckets, d_model, padding_idx=0)
                        for d in self.seq_domains
                    })

        # ================== Delta-t Bucket Embedding (W2.7, per-domain, optional) ==================
        if num_delta_buckets > 0:
            self.delta_embeddings = nn.ModuleDict({
                d: nn.Embedding(num_delta_buckets, d_model, padding_idx=0)
                for d in self.seq_domains
            })

        # ================== HyFormer Components ==================
        if use_time_summary_features:
            time_summary_dim = self.num_sequences * 8
            self.time_summary_proj = nn.Sequential(
                nn.Linear(time_summary_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
        if use_seq_overflow_summary_features:
            overflow_summary_dim = self.num_sequences * self.seq_overflow_summary_features_per_domain
            self.seq_overflow_summary_proj = nn.Sequential(
                nn.Linear(overflow_summary_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )

        # MultiSeqQueryGenerator
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
        )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                longer_gather_side=seq_longer_gather_side,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection for the original HyFormer head.  It is still
        # available when use_ui_attn_fusion=False, and kept for backwards
        # compatibility with old checkpoints.
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # UI-attention fusion head (main-fusion version, not a residual side path).
        #   u_pool: mean pooled user-side NS tokens (user int + user dense)
        #   i_pool: mean pooled item-side NS tokens (item int + item dense)
        #   i_attn: i_pool attends over all sequence tokens from all domains
        # final representation = concat(u_pool, i_pool, i_attn).
        if self.use_ui_attn_fusion:
            self.ui_attn_q_norm = nn.LayerNorm(d_model)
            self.ui_attn_k_norm = nn.LayerNorm(d_model)
            self.ui_attn_q = nn.Linear(d_model, d_model, bias=False)
            self.ui_attn_k = nn.Linear(d_model, d_model, bias=False)
            self.ui_attn_v = nn.Linear(d_model, d_model, bias=False)
            self.ui_attn_out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            )
            classifier_in_dim = 3 * d_model
        else:
            classifier_in_dim = d_model

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(classifier_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            if self.per_domain_time_embeddings:
                for emb in self.time_embeddings.values():
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
            else:
                nn.init.xavier_normal_(self.time_embedding.weight.data)
                self.time_embedding.weight.data[0, :] = 0
                if self.domain_time_residual_embeddings:
                    for emb in self.time_residual_embeddings.values():
                        nn.init.zeros_(emb.weight.data)

        if self.num_delta_buckets > 0:
            for emb in self.delta_embeddings.values():
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.use_seq_periodic_time_features:
            if self.per_domain_seq_periodic_time_features:
                periodic_embs = []
                if self.use_seq_hour_of_day_feature:
                    periodic_embs.extend(self.seq_hour_embeddings.values())
                if self.use_seq_day_of_week_feature:
                    periodic_embs.extend(self.seq_dow_embeddings.values())
            else:
                periodic_embs = []
                if self.use_seq_hour_of_day_feature:
                    periodic_embs.append(self.seq_hour_embedding)
                if self.use_seq_day_of_week_feature:
                    periodic_embs.append(self.seq_dow_embedding)
            for emb in periodic_embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += len(self.time_embeddings) if self.per_domain_time_embeddings else 1
            if self.domain_time_residual_embeddings and not self.per_domain_time_embeddings:
                skip_count += len(self.time_residual_embeddings)
        # delta_embeddings always preserved (W2.7, per-domain)
        if self.num_delta_buckets > 0:
            skip_count += len(self.delta_embeddings)
        if self.use_seq_periodic_time_features:
            periodic_feature_count = (
                int(self.use_seq_hour_of_day_feature)
                + int(self.use_seq_day_of_week_feature)
            )
            skip_count += (
                len(self.seq_domains) * periodic_feature_count
                if self.per_domain_seq_periodic_time_features
                else periodic_feature_count
            )

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        delta_bucket_ids: torch.Tensor,
        hour_bucket_ids: torch.Tensor,
        dow_bucket_ids: torch.Tensor,
        domain_name: str,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        if self.use_seq_periodic_time_features:
            if self.per_domain_seq_periodic_time_features:
                if self.use_seq_hour_of_day_feature:
                    emb_list.append(self.seq_hour_embeddings[domain_name](hour_bucket_ids))
                if self.use_seq_day_of_week_feature:
                    emb_list.append(self.seq_dow_embeddings[domain_name](dow_bucket_ids))
            else:
                if self.use_seq_hour_of_day_feature:
                    emb_list.append(self.seq_hour_embedding(hour_bucket_ids))
                if self.use_seq_day_of_week_feature:
                    emb_list.append(self.seq_dow_embedding(dow_bucket_ids))
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            if self.per_domain_time_embeddings:
                token_emb = token_emb + self.time_embeddings[domain_name](time_bucket_ids)
            else:
                token_emb = token_emb + self.time_embedding(time_bucket_ids)
                if self.domain_time_residual_embeddings:
                    token_emb = token_emb + self.time_residual_embeddings[domain_name](
                        time_bucket_ids)

        # Add delta-t bucket embedding (W2.7, per-domain; padding_idx=0 zeros out
        # padding positions and the last token of each seq, which has no neighbor)
        if self.num_delta_buckets > 0:
            token_emb = token_emb + self.delta_embeddings[domain_name](delta_bucket_ids)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        return_tokens: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list, list]]:
        """Runs the multi-sequence block stack.

        When ``return_tokens`` is True, returns the final NS tokens and encoded
        sequence tokens instead of the original flattened-Q projection.  This is
        used by the UI-attention fusion head.
        """
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
            )

        if return_tokens:
            return curr_ns, curr_seqs, curr_masks

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def _pool_ns_span(
        self, ns_tokens: torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """Mean-pool a contiguous span of NS tokens.

        The span is non-empty for the active TAAC config, but the fallback keeps
        the method safe for unusual schemas.
        """
        if end <= start:
            return ns_tokens.new_zeros(ns_tokens.shape[0], self.d_model)
        return ns_tokens[:, start:end, :].mean(dim=1)

    def _item_to_sequence_attention(
        self,
        i_pool: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
    ) -> torch.Tensor:
        """Use pooled item token as query to attend over all sequence tokens.

        Args:
            i_pool: (B, D) pooled item representation.
            seq_tokens_list: list of (B, L_d, D) encoded sequence tokens.
            seq_masks_list: list of (B, L_d), True means padding.

        Returns:
            i_attn: (B, D), item-conditioned history representation.
        """
        seq_tokens = torch.cat(seq_tokens_list, dim=1)  # (B, L_total, D)
        seq_mask = torch.cat(seq_masks_list, dim=1)      # (B, L_total)

        q = self.ui_attn_q(self.ui_attn_q_norm(i_pool)).unsqueeze(1)  # (B, 1, D)
        k = self.ui_attn_k(self.ui_attn_k_norm(seq_tokens))           # (B, L, D)
        v = self.ui_attn_v(seq_tokens)                                # (B, L, D)

        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        scores = scores.masked_fill(seq_mask.unsqueeze(1), -1e4)

        attn = F.softmax(scores, dim=-1)
        valid = (~seq_mask).unsqueeze(1).to(attn.dtype)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        ctx = torch.matmul(attn, v).squeeze(1)  # (B, D)
        return self.ui_attn_out(ctx)

    def _build_ui_attn_fusion(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
    ) -> torch.Tensor:
        """Build concat(u_pool, i_pool, i_attn) for the final classifier."""
        u_pool = self._pool_ns_span(ns_tokens, self.user_token_start, self.user_token_end)
        i_pool = self._pool_ns_span(ns_tokens, self.item_token_start, self.item_token_end)
        i_attn = self._item_to_sequence_attention(i_pool, seq_tokens_list, seq_masks_list)
        return torch.cat([u_pool, i_pool, i_attn], dim=-1)

    def _project_user_dense_tokens(self, user_dense_feats: torch.Tensor) -> torch.Tensor:
        if isinstance(self.user_dense_proj, DenseGroupProjector):
            return self.user_dense_proj(user_dense_feats)
        return F.silu(self.user_dense_proj(user_dense_feats)).unsqueeze(1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = self._project_user_dense_tokens(inputs.user_dense_feats)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)
        if self.use_time_summary_features:
            time_summary_tok = self.time_summary_proj(inputs.time_summary_feats).unsqueeze(1)
            ns_parts.append(time_summary_tok)
        if self.use_seq_overflow_summary_features:
            overflow_summary_tok = self.seq_overflow_summary_proj(
                inputs.seq_overflow_summary_feats
            ).unsqueeze(1)
            ns_parts.append(overflow_summary_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_delta_buckets[domain],
                inputs.seq_hour_buckets[domain],
                inputs.seq_dow_buckets[domain],
                domain)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + final representation
        if self.use_ui_attn_fusion:
            curr_ns, curr_seqs, curr_masks = self._run_multi_seq_blocks(
                q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                apply_dropout=self.training,
                return_tokens=True,
            )
            output = self._build_ui_attn_fusion(curr_ns, curr_seqs, curr_masks)
        else:
            output = self._run_multi_seq_blocks(
                q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                apply_dropout=self.training
            )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = self._project_user_dense_tokens(inputs.user_dense_feats)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)
        if self.use_time_summary_features:
            time_summary_tok = self.time_summary_proj(inputs.time_summary_feats).unsqueeze(1)
            ns_parts.append(time_summary_tok)
        if self.use_seq_overflow_summary_features:
            overflow_summary_tok = self.seq_overflow_summary_proj(
                inputs.seq_overflow_summary_feats
            ).unsqueeze(1)
            ns_parts.append(overflow_summary_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                inputs.seq_delta_buckets[domain],
                inputs.seq_hour_buckets[domain],
                inputs.seq_dow_buckets[domain],
                domain)
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        q_tokens_list = self.query_generator(ns_tokens, seq_tokens_list, seq_masks_list)

        if self.use_ui_attn_fusion:
            curr_ns, curr_seqs, curr_masks = self._run_multi_seq_blocks(
                q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                apply_dropout=False,
                return_tokens=True,
            )
            output = self._build_ui_attn_fusion(curr_ns, curr_seqs, curr_masks)
        else:
            output = self._run_multi_seq_blocks(
                q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
                apply_dropout=False
            )

        logits = self.clsfier(output)
        return logits, output
