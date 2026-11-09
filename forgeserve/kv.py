"""KV-cache views. A "view" is what one attention layer sees: it stores the new keys and values of a
step and returns every key and value so far, for each batch row, at absolute positions 0..S-1
(rows may be padded at the end; the attention mask hides the padding)."""

from typing import Optional, Tuple

import torch


class ContiguousKV:
    """Plain per-layer cache for rows that all have the same length."""

    def __init__(self):
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat([self.k, k], dim=2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def truncate(self, n: int) -> None:
        """Keep only the first n positions (used by speculative decoding to drop rejected tokens)."""
        if self.k is not None:
            self.k, self.v = self.k[:, :, :n], self.v[:, :, :n]


def new_contiguous_cache(n_layers: int):
    return [ContiguousKV() for _ in range(n_layers)]
