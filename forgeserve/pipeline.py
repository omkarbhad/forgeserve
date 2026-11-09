"""Pipeline parallelism: split the layers into consecutive stages, one process per stage.

Stage 0 holds the embeddings, the last stage holds the final norm and the output head, and every stage holds
its own slice of the layers and of the KV cache. The batch is cut into micro-batches that flow through the
stages one after another, so while stage 1 works on micro-batch 0, stage 0 already starts micro-batch 1
(as in GPipe, https://arxiv.org/abs/1811.06965). Activations move with point-to-point send/recv.

With p stages and m micro-batches, a fraction (p - 1) / (m + p - 1) of stage time is idle ("the bubble");
see `bubble_fraction`. This implementation shows the data flow and checks it for exactness on CPU.
It does not claim speed-ups, because the stages here share one machine.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from .config import ModelConfig
from .kv import ContiguousKV
from .model import Block, RMSNorm, precompute_rope
from .parallel import SINGLE
from .sampling import sample


def stage_layers(n_layers: int, size: int) -> List[Tuple[int, int]]:
    """Consecutive layer ranges [lo, hi), as even as possible (earlier stages take the extra layers)."""
    if size > n_layers:
        raise ValueError("more pipeline stages than layers")
    base, extra = divmod(n_layers, size)
    out, lo = [], 0
    for r in range(size):
        hi = lo + base + (1 if r < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


def bubble_fraction(stages: int, microbatches: int) -> float:
    """Share of stage time wasted waiting for the pipeline to fill and drain."""
    return (stages - 1) / (microbatches + stages - 1)


class PipelineStage(nn.Module):
    def __init__(self, cfg: ModelConfig, rank: int, size: int):
        super().__init__()
        self.cfg, self.rank, self.size = cfg, rank, size
        self.lo, self.hi = stage_layers(cfg.n_layers, size)[rank]
        self.is_first, self.is_last = rank == 0, rank == size - 1
        self.layers = nn.ModuleList(Block(cfg, SINGLE) for _ in range(self.hi - self.lo))
        if self.is_first:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if self.is_last:
            self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)   # a copy of the tied embedding
        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def load_from_full(self, full: Dict[str, torch.Tensor]) -> None:
        """Take this stage's weights out of a full model's state dict."""
        own = {}
        for name, w in full.items():
            if name.startswith("layers."):
                i = int(name.split(".")[1])
                if self.lo <= i < self.hi:
                    own[name.replace(f"layers.{i}.", f"layers.{i - self.lo}.", 1)] = w
            elif name == "tok_emb.weight" and self.is_first:
                own[name] = w
            elif name == "norm.weight" and self.is_last:
                own[name] = w
            elif name == "lm_head.weight" and self.is_last:
                own[name] = w
        self.load_state_dict(own)

    @torch.no_grad()
    def run_layers(self, x, positions, caches: List[ContiguousKV]):
        for layer, kv in zip(self.layers, caches):
            x = layer(x, self.rope_cos, self.rope_sin, positions, kv)
        return x


class PipelineState:
    """One KV cache per local layer for each micro-batch (each micro-batch is a fixed set of batch rows)."""

    def __init__(self, stage: PipelineStage, batch: int, microbatches: int):
        self.sizes = [len(c) for c in torch.arange(batch).tensor_split(microbatches)]
        self.caches = [[ContiguousKV() for _ in stage.layers] for _ in self.sizes]

    @property
    def length(self) -> int:
        return self.caches[0][0].length if self.caches[0] else 0


@torch.no_grad()
def pipeline_forward(stage: PipelineStage, state: PipelineState, idx: Optional[torch.Tensor], t: int):
    """Push t new tokens per row through the pipeline. `idx` (B, t) is needed on stage 0 only.

    Returns the logits (B, t, vocab) on the last stage and None elsewhere."""
    start = state.length
    positions_full = start + torch.arange(t)
    sends, outputs, row = [], [], 0
    for mb, size in enumerate(state.sizes):
        positions = positions_full[None, :].expand(size, t)
        if stage.is_first:
            x = stage.tok_emb(idx[row : row + size])
        else:
            x = torch.empty(size, t, stage.cfg.d_model)
            dist.recv(x, src=stage.rank - 1)
        x = stage.run_layers(x, positions, state.caches[mb])
        if stage.is_last:
            outputs.append(stage.lm_head(stage.norm(x)))
        else:
            sends.append((dist.isend(x.contiguous(), dst=stage.rank + 1), x))   # keep x alive until the send is done
        row += size
    for work, _ in sends:
        work.wait()
    return torch.cat(outputs, dim=0) if stage.is_last else None


@torch.no_grad()
def pipeline_generate(stage: PipelineStage, prompt: torch.Tensor, max_new_tokens: int, microbatches: int = 1,
                      temperature: float = 0.0, generator=None) -> torch.Tensor:
    """Generate with the pipeline. Every stage calls this with the same `prompt` (B, T) and gets the same result.

    The last stage samples; the chosen tokens are broadcast so stage 0 can feed them back in."""
    b, t = prompt.shape
    max_new_tokens = min(max_new_tokens, stage.cfg.max_seq_len - t)
    state = PipelineState(stage, b, microbatches)
    out, idx, new = prompt, prompt, t
    for _ in range(max_new_tokens):
        logits = pipeline_forward(stage, state, idx if stage.is_first else None, new)
        nxt = sample(logits[:, -1], temperature, generator) if stage.is_last else torch.empty(b, dtype=torch.long)
        dist.broadcast(nxt, src=stage.size - 1)
        out = torch.cat([out, nxt[:, None]], dim=1)
        idx, new = nxt[:, None], 1
    return out
