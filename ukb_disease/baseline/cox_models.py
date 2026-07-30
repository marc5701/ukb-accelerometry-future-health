"""Sequence-head architectures over N-day windows of (AR + HA) per-day means.

All three heads share the forward signature:

    forward(window: (B, N, in_dim), valid_mask: (B, N) bool,
            covariates: (B, n_cov)) -> hazards: (B, n_phecodes)

where `in_dim = 512` (AR 256-d concat HA 256-d). `valid_mask` is True for
real day positions; padded positions are False. In the WindowDataset's
default mode, windows are constrained to N consecutive valid days, so the
mask is all-True at runtime, but mask handling is implemented anyway so
the architecture can later admit partial-mask windows without re-coding.

Architectures:
- FlatMLPHead         : concat all N day vectors, 2-layer MLP head
- LSTMSeqHead         : input projection, biLSTM, masked mean pool
- TransformerSeqHead  : input projection, transformer encoder with a [CLS]
                        token, CLS pool.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: drops the entire residual contribution with prob p.

    Standard ViT-style implementation: at training time, with probability
    `p` the residual contribution is zeroed; surviving samples are scaled by
    `1/(1-p)` so the expected output magnitude is unchanged. At eval time,
    pass-through.
    """

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class _DropPathTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer with DropPath wrapping both residual sub-blocks.

    Subclass override pattern: `_sa_block` and `_ff_block` produce the
    self-attention and FFN contributions added back to the residual stream.
    Wrapping their outputs with DropPath makes the residual stochastic.
    Works for both norm_first=True and norm_first=False; this codebase uses
    norm_first=True (pre-norm).
    """

    def __init__(self, *args, drop_path: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_path = DropPath(drop_path)

    def _sa_block(self, *args, **kwargs):
        return self.drop_path(super()._sa_block(*args, **kwargs))

    def _ff_block(self, *args, **kwargs):
        return self.drop_path(super()._ff_block(*args, **kwargs))


def _make_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Standard transformer sinusoidal positional encoding table.

    Returns a (max_len, d_model) tensor with the canonical sin/cos pattern.
    Parameter-free; meant to be registered as a buffer.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class _FactorizedOutput(nn.Module):
    """Low-rank factorized hazard projection: Linear(h, r, bias=False) then Linear(r, n_out).

    Diagnostic: tests whether the (flat) hazard outputs live on a rank-r subject
    manifold. The second Linear's (n_out, r) loadings + bias give per-disease
    intercepts for free. Operates on the FLAT output width (n_phecodes * n_bins),
    so the discrete-time reshape downstream is unaffected.
    """

    def __init__(self, hidden_dim: int, n_out: int, rank: int):
        super().__init__()
        self.U = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.U(x))


class _HeadMixin:
    """Helper for the final linear hazard head, shared across architectures."""

    def _make_head(
        self,
        pooled_dim: int,
        n_phecodes: int,
        n_covar: int,
        hidden_dim: int,
        dropout: float,
        output_rank: int = 0,
    ) -> nn.Module:
        # output_rank == 0 keeps the dense Linear (the default configuration).
        final = (
            _FactorizedOutput(hidden_dim, n_phecodes, output_rank)
            if output_rank and output_rank > 0
            else nn.Linear(hidden_dim, n_phecodes)
        )
        return nn.Sequential(
            nn.Linear(pooled_dim + n_covar, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            final,
        )


class FlatMLPHead(nn.Module, _HeadMixin):
    """Baseline: concat N day vectors, then an MLP.

    Tests whether the lift comes from more day-units as input versus a
    subject-level mean. If FlatMLP beats the baseline mean pool, the lift exists
    at the input granularity level (no temporal modeling needed).
    """

    def __init__(
        self,
        window_size: int,
        in_dim: int = 512,
        n_phecodes: int = 289,
        n_covar: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        input_norm: bool = False,
        output_rank: int = 0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.input_norm = nn.LayerNorm(self.in_dim) if input_norm else nn.Identity()
        flat_dim = self.window_size * self.in_dim
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = self._make_head(hidden_dim, n_phecodes, n_covar, hidden_dim, dropout,
                                    output_rank=output_rank)

    def forward(self, window: torch.Tensor, valid_mask: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        B, N, _ = window.shape
        window = self.input_norm(window)
        # Zero out padded days before flatten (defensive; usually all-True at runtime)
        window = window * valid_mask.unsqueeze(-1).float()
        flat = window.reshape(B, N * self.in_dim)
        pooled = self.proj(flat)
        return self.head(torch.cat([pooled, covariates], dim=-1))


class LSTMSeqHead(nn.Module, _HeadMixin):
    """Bidirectional LSTM over the N-day sequence with masked mean pool.

    Captures sequence dynamics across days. 1-layer biLSTM, hidden=128 per
    direction gives a 256-d output, mean over valid timesteps, then the
    standard hazard head.
    """

    def __init__(
        self,
        window_size: int,
        in_dim: int = 512,
        n_phecodes: int = 289,
        n_covar: int = 2,
        hidden_dim: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
        input_norm: bool = False,
        output_rank: int = 0,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        self.input_proj = nn.Linear(in_dim, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Bi-LSTM output dim = 2 * hidden_dim
        self.lstm_out_dim = 2 * hidden_dim
        self.head = self._make_head(self.lstm_out_dim, n_phecodes, n_covar, head_hidden, dropout,
                                    output_rank=output_rank)

    def forward(self, window: torch.Tensor, valid_mask: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        # window: (B, N, in_dim)
        x = self.layer_norm(self.input_proj(self.input_norm(window)))
        out, _ = self.lstm(x)  # (B, N, 2*hidden)
        # Masked mean pool
        valid = valid_mask.float().unsqueeze(-1)  # (B, N, 1)
        pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(torch.cat([pooled, covariates], dim=-1))


class TransformerSeqHead(nn.Module, _HeadMixin):
    """Transformer encoder over the N-day sequence with [CLS] pooling.

    Pre-norm, default 2-layer / 4-head / learned position embeddings
    (max_days=8 covers N in {3,5,7}). Optional pe_type='sinusoidal'
    swaps to parameter-free sinusoidal PE; optional drop_path > 0 adds
    stochastic-depth on both residual paths of each encoder layer.
    """

    def __init__(
        self,
        window_size: int,
        in_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        n_phecodes: int = 289,
        n_covar: int = 2,
        head_hidden: int = 256,
        dropout: float = 0.1,
        max_days: int = 8,
        pe_type: str = "learned",
        drop_path: float = 0.0,
        input_norm: bool = False,
        output_rank: int = 0,
        cov_bottleneck_dim: int = 0,
        cov_idx=None,
        stream_split: int = 0,
        across_day_pool: str = "cls",
        pma_seeds_day: int = 1,
        aux_age: bool = False,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.stream_split = int(stream_split)
        self.max_days = int(max_days)
        self.pe_type = str(pe_type)
        # Covariance bottleneck (split input projection). When
        # cov_bottleneck_dim > 0 and cov_idx names the within-day covariance
        # columns (interleaved: last C of each modality block), the base
        # mean-std channels keep the full-rank input_proj while the cov tail
        # gets a low-rank Linear(cov,k)->Linear(k,d_model) bottleneck added in.
        # This capacity-limits the off-diagonal so it can't distort the
        # marginals. cov_bottleneck_dim == 0 uses the single input_proj path.
        self.cov_bottleneck_dim = int(cov_bottleneck_dim)
        if self.cov_bottleneck_dim > 0 and cov_idx is not None and len(cov_idx) > 0:
            cov_set = set(int(i) for i in cov_idx)
            cov_idx_t = torch.as_tensor(sorted(cov_set), dtype=torch.long)
            base_idx_t = torch.as_tensor(
                [i for i in range(self.in_dim) if i not in cov_set], dtype=torch.long
            )
            self.register_buffer("cov_idx", cov_idx_t, persistent=True)
            self.register_buffer("base_idx", base_idx_t, persistent=True)
            base_dim = int(base_idx_t.numel())
            cov_dim = int(cov_idx_t.numel())
            self.input_norm = nn.LayerNorm(base_dim) if input_norm else nn.Identity()
            self.input_norm_cov = nn.LayerNorm(cov_dim) if input_norm else nn.Identity()
            self.input_proj = nn.Linear(base_dim, d_model)
            self.cov_down = nn.Linear(cov_dim, self.cov_bottleneck_dim, bias=False)
            self.cov_up = nn.Linear(self.cov_bottleneck_dim, d_model)
        elif self.stream_split > 0:
            # Two-tower: split the input at `stream_split` (AR block | HA block) and
            # give each stream its own LayerNorm + Linear to d_model//2, concatenated
            # back to d_model. Protects the weaker AR stream from being crushed in a
            # single shared in_dim->d_model projection. cov_bottleneck stays off.
            self.cov_bottleneck_dim = 0
            d_ar = d_model // 2
            d_ha = d_model - d_ar
            ar_dim = self.stream_split
            ha_dim = self.in_dim - self.stream_split
            self.input_norm = nn.LayerNorm(ar_dim) if input_norm else nn.Identity()
            self.input_norm_ha = nn.LayerNorm(ha_dim) if input_norm else nn.Identity()
            self.input_proj = nn.Linear(ar_dim, d_ar)
            self.input_proj_ha = nn.Linear(ha_dim, d_ha)
        else:
            self.cov_bottleneck_dim = 0
            self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
            self.input_proj = nn.Linear(in_dim, d_model)
        # +1 for the CLS slot at position 0
        if self.pe_type == "learned":
            self.pos_emb = nn.Embedding(self.max_days + 1, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        elif self.pe_type == "sinusoidal":
            self.register_buffer(
                "pos_emb_table", _make_sinusoidal_pe(self.max_days + 1, d_model)
            )
        else:
            raise ValueError(
                f"Unknown pe_type '{self.pe_type}'. Choices: learned, sinusoidal"
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        encoder_layer = _DropPathTransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            drop_path=drop_path,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # Learnable across-day pooling. "cls" (default) reads the CLS token;
        # "pma" replaces the CLS read with a masked PMA over the N real day
        # tokens (k seeds, mean-reduced to keep head width fixed).
        self.across_day_pool = str(across_day_pool)
        if self.across_day_pool == "pma":
            from ukb_disease.pool_attention.set_transformer import PMA
            self.day_pma = PMA(d_model, num_heads=n_heads, num_seeds=int(pma_seeds_day),
                               ff_mult=ff_mult, dropout=dropout)
        elif self.across_day_pool != "cls":
            raise ValueError(f"across_day_pool must be 'cls' or 'pma', got {across_day_pool!r}")
        self.head = self._make_head(d_model, n_phecodes, n_covar, head_hidden, dropout,
                                    output_rank=output_rank)
        # Optional aux-age regression head predicting per-subject ar_emb from the
        # pooled trunk (shapes the shared representation; dropped at inference). The covariate
        # vector arrives one wider (last col = ar_emb target) but we slice it to n_covar for the
        # disease head, so cov_in is unchanged when aux_age is off.
        self._head_n_covar = int(n_covar)
        self.aux_age = bool(aux_age)
        if self.aux_age:
            self.aux_age_head = nn.Linear(d_model, 1)

    def forward(self, window: torch.Tensor, valid_mask: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        B, N, _ = window.shape
        if N > self.max_days:
            raise ValueError(
                f"TransformerSeqHead: input N={N} > max_days={self.max_days}"
            )
        if self.cov_bottleneck_dim > 0:
            base = window.index_select(-1, self.base_idx)
            cov = window.index_select(-1, self.cov_idx)
            x = self.input_proj(self.input_norm(base)) + self.cov_up(
                self.cov_down(self.input_norm_cov(cov))
            )  # (B, N, d_model)
        elif self.stream_split > 0:
            ar = window[..., : self.stream_split]
            ha = window[..., self.stream_split:]
            x = torch.cat(
                [self.input_proj(self.input_norm(ar)),
                 self.input_proj_ha(self.input_norm_ha(ha))], dim=-1
            )  # (B, N, d_model)
        else:
            x = self.input_proj(self.input_norm(window))  # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, x], dim=1)  # (B, 1+N, d_model)
        pos_idx = torch.arange(1 + N, device=window.device)
        if self.pe_type == "learned":
            pos = self.pos_emb(pos_idx)
        else:
            pos = self.pos_emb_table[pos_idx]
        seq = seq + pos.unsqueeze(0)
        cls_real = torch.ones(B, 1, dtype=torch.bool, device=valid_mask.device)
        full_mask = torch.cat([cls_real, valid_mask], dim=1)  # True = real
        key_padding_mask = ~full_mask  # nn.Transformer wants True = pad
        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        if self.across_day_pool == "pma":
            # PMA over the N real day tokens (drop CLS at index 0), masked by valid days.
            pooled = self.day_pma(out[:, 1:, :], key_padding_mask=~valid_mask).mean(dim=1)
        else:
            pooled = out[:, 0, :]  # CLS
        cov_in = covariates[..., :self._head_n_covar]   # slice off the appended ar_emb col (no-op when off)
        out_head = self.head(torch.cat([pooled, cov_in], dim=-1))
        # Append the ar_emb prediction ONLY in training mode (it shapes the trunk via its
        # gradient); at eval/inference the head is dropped so scoring buffers stay n_total wide.
        if self.aux_age and self.training:
            out_head = torch.cat([out_head, self.aux_age_head(pooled)], dim=-1)
        return out_head


class MambaSeqHead(nn.Module, _HeadMixin):
    """Selective-SSM (kernel-free minGRU/Mamba-family) over the N-day sequence.

    A day-level SSM sanity check. With only 3-7 timesteps this is far too short
    for an SSM to beat the transformer, so a tie is expected; the patch-level
    variant is where a state-space model can matter. Bidirectional selective
    recurrence plus masked-mean pool, same forward contract as the other heads.
    """

    def __init__(
        self,
        window_size: int,
        in_dim: int = 512,
        d_model: int = 256,
        n_layers: int = 2,
        n_phecodes: int = 289,
        n_covar: int = 2,
        head_hidden: int = 256,
        dropout: float = 0.1,
        input_norm: bool = False,
        output_rank: int = 0,
    ):
        super().__init__()
        from ukb_disease.baseline.mamba_block import MambaBlock, MaskedMeanPool
        self.window_size = int(window_size)
        self.in_dim = int(in_dim)
        self.d_model = int(d_model)
        self.input_norm = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, expand=2, conv_kernel=2, causal=False, dropout=dropout)
            for _ in range(int(n_layers))
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = MaskedMeanPool()
        self.head = self._make_head(d_model, n_phecodes, n_covar, head_hidden, dropout,
                                    output_rank=output_rank)

    def forward(self, window: torch.Tensor, valid_mask: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(self.input_norm(window))
        for blk in self.blocks:
            x = blk(x, valid_mask)
        x = self.norm(x)
        pooled = self.pool(x, valid_mask)
        return self.head(torch.cat([pooled, covariates], dim=-1))


_ARCH_REGISTRY = {
    "mlp_flat": FlatMLPHead,
    "lstm": LSTMSeqHead,
    "transformer": TransformerSeqHead,
    "mamba": MambaSeqHead,
}


def build_cox_model(
    arch: str,
    window_size: int,
    n_phecodes: int,
    n_covar: int,
    in_dim: int = 512,
    dropout: float = 0.1,
    n_layers: int = 2,
    pe_type: str = "learned",
    drop_path: float = 0.0,
    n_out_per_phecode: int = 1,
    input_norm: bool = False,
    output_rank: int = 0,
    max_days: int = 8,
    cov_bottleneck_dim: int = 0,
    cov_idx=None,
    stream_split: int = 0,
    across_day_pool: str = "cls",
    pma_seeds_day: int = 1,
    aux_age: bool = False,
) -> nn.Module:
    """Factory: instantiate the sequence-head model by name.

    Defaults are conservative (small models): MLP ~135k params, LSTM ~600k,
    Transformer ~800k. Tested for 3-7 day windows.

    `n_layers`, `pe_type`, and `drop_path` only apply to the transformer
    arch (silently ignored for mlp_flat / lstm).

    `n_out_per_phecode` widens the final head to `n_phecodes * n_out_per_phecode`
    outputs (discrete-time head: one logit per (phecode, time-bin); default 1 =
    Cox single-hazard). The caller reshapes the flat output to (B, n_phecodes,
    n_out_per_phecode). `input_norm` inserts a LayerNorm over the input feature
    dim (distributional pooling: tames the mean-std scale mismatch); default
    off leaves the Cox/mean path unchanged.
    """
    if arch not in _ARCH_REGISTRY:
        raise ValueError(f"Unknown sequence-head arch '{arch}'. Choices: {list(_ARCH_REGISTRY)}")
    cls = _ARCH_REGISTRY[arch]
    n_head_out = int(n_phecodes) * int(n_out_per_phecode)
    kwargs = dict(
        window_size=window_size,
        in_dim=in_dim,
        n_phecodes=n_head_out,
        n_covar=n_covar,
        dropout=dropout,
        input_norm=input_norm,
        output_rank=output_rank,
    )
    if arch == "transformer":
        kwargs["n_layers"] = n_layers
        kwargs["pe_type"] = pe_type
        kwargs["drop_path"] = drop_path
        kwargs["max_days"] = max_days
        kwargs["cov_bottleneck_dim"] = cov_bottleneck_dim
        kwargs["cov_idx"] = cov_idx
        kwargs["stream_split"] = stream_split
        kwargs["across_day_pool"] = across_day_pool
        kwargs["pma_seeds_day"] = pma_seeds_day
        kwargs["aux_age"] = aux_age
    elif arch == "mamba":
        kwargs["n_layers"] = n_layers
    return cls(**kwargs)


__all__ = [
    "DropPath",
    "FlatMLPHead",
    "LSTMSeqHead",
    "TransformerSeqHead",
    "MambaSeqHead",
    "build_cox_model",
]
