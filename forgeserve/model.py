"""Decoder-only transformer (LLaMA style) written for inference at scale.

RMSNorm -> grouped-query attention with rotary embeddings -> SwiGLU, tied embeddings. Three things make it
usable by the parallel and serving code in this package:

- Positions are explicit: every row of a batch can sit at a different position, which continuous batching needs.
- The KV cache is a "view" with an `update(k, v)` method, so the same model runs on a plain cache or a paged one.
- Tensor parallelism is built in: with a `TPContext` of size N every layer holds 1/N of its weights, and the
  two row-parallel projections (attention output, MLP down) sum their partial results across shards.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .parallel import SINGLE, TPContext


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, D); positions: (B, T) absolute positions, so each row may be at its own position."""
    c = cos[positions][:, None].to(x.dtype)   # (B, 1, T, D/2)
    s = sin[positions][:, None].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Attention(nn.Module):
    """Column-parallel q/k/v (a shard owns whole heads), row-parallel output projection."""

    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.tp = tp
        self.hd = cfg.head_dim
        self.nh, self.nkv = cfg.n_heads // tp.size, cfg.n_kv_heads // tp.size   # heads held by this shard
        self.rep = cfg.n_heads // cfg.n_kv_heads
        self.wq = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, positions, kv):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin, positions), apply_rope(k, cos, sin, positions)

        k_all, v_all = kv.update(k, v)                        # (B, nkv, S, D): keys at positions 0..S-1
        k_all = k_all.repeat_interleave(self.rep, dim=1)
        v_all = v_all.repeat_interleave(self.rep, dim=1)

        s = k_all.shape[2]
        # A query at position p may see keys 0..p. This also hides padding keys, which sit beyond every valid p.
        mask = (torch.arange(s, device=x.device)[None, None, :] <= positions[:, :, None])[:, None]
        out = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)
        out = self.wo(out.transpose(1, 2).reshape(b, t, -1))   # partial sum when sharded
        return self.tp.all_reduce(out)


class SwiGLU(nn.Module):
    """Column-parallel gate/up, row-parallel down."""

    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.tp = tp
        hidden = cfg.ffn_dim // tp.size
        self.w_gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.tp.all_reduce(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, tp: TPContext):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, tp)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg, tp)

    def forward(self, x, cos, sin, positions, kv):
        x = x + self.attn(self.attn_norm(x), cos, sin, positions, kv)
        return x + self.mlp(self.mlp_norm(x))


class ForgeModel(nn.Module):
    """Embeddings and the output head are replicated on every shard (vocabulary-parallel is not implemented)."""

    def __init__(self, cfg: ModelConfig, tp: TPContext = SINGLE):
        super().__init__()
        cfg.check_tensor_parallel(tp.size)
        self.cfg, self.tp = cfg, tp
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, tp) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def embed(self, idx):
        return self.tok_emb(idx)

    def head(self, x):
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def forward(self, idx, positions: Optional[torch.Tensor] = None, kv: Optional[List] = None):
        """idx: (B, T) token ids -> logits (B, T, vocab).

        positions: (B, T) absolute positions; default continues after whatever `kv` already holds.
        kv: one cache view per layer (see kv.py); None runs without a cache.
        """
        b, t = idx.shape
        if kv is None:
            from .kv import new_contiguous_cache
            kv = new_contiguous_cache(self.cfg.n_layers)
        if positions is None:
            start = kv[0].length
            positions = (start + torch.arange(t, device=idx.device))[None, :].expand(b, t)
        if int(positions.max()) >= self.cfg.max_seq_len:
            raise ValueError(f"position {int(positions.max())} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.embed(idx)
        for layer, view in zip(self.layers, kv):
            x = layer(x, self.rope_cos, self.rope_sin, positions, view)
        return self.head(x)
