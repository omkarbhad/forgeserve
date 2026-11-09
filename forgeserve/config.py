"""Model configuration."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4           # grouped-query attention
    ffn_dim: Optional[int] = None  # default: 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = (int(8 / 3 * self.d_model) + 63) // 64 * 64
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def check_tensor_parallel(self, size: int) -> None:
        for name, n in (("n_heads", self.n_heads), ("n_kv_heads", self.n_kv_heads), ("ffn_dim", self.ffn_dim)):
            if n % size:
                raise ValueError(f"{name} ({n}) must be divisible by the tensor-parallel size ({size})")

    def to_dict(self) -> dict:
        return asdict(self)
