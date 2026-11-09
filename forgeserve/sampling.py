"""Token sampling and a plain reference generator."""

import torch
import torch.nn.functional as F

from .kv import new_contiguous_cache


def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax of logits / temperature. temperature 0 gives a one-hot on the argmax."""
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return F.softmax(logits.float() / temperature, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0, generator=None) -> torch.Tensor:
    """logits (..., V) -> token ids (...,)."""
    if temperature == 0:
        return logits.argmax(-1)
    probs = probs_from_logits(logits, temperature)
    return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1, generator=generator).reshape(probs.shape[:-1])


@torch.no_grad()
def generate(model, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 0.0, generator=None):
    """Reference generator for rows of equal length: prefill, then one token at a time with a plain cache."""
    kv = new_contiguous_cache(model.cfg.n_layers)
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len - prompt.shape[1])
    out = prompt
    logits = model(prompt, kv=kv)
    for _ in range(max_new_tokens):
        nxt = sample(logits[:, -1], temperature, generator)[:, None]
        out = torch.cat([out, nxt], dim=1)
        logits = model(nxt, kv=kv)
    return out
