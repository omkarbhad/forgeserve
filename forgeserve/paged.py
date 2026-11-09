"""Paged KV cache (vLLM PagedAttention, https://arxiv.org/abs/2309.06180).

Instead of reserving max_seq_len slots per sequence, the cache is one pool of fixed-size blocks. A sequence
owns a list of blocks (its block table) that need not be contiguous in the pool, and takes a new block only
when the previous one is full. So memory is wasted only in the last block of each sequence, and many
sequences of different lengths can share the pool.

This is the gather formulation in plain PyTorch: blocks are looked up through the block table and the
sequences' keys and values are laid out per row before ordinary attention. There is no fused kernel.
"""

from typing import List, Tuple

import torch

from .config import ModelConfig
from .parallel import SINGLE, TPContext


class OutOfBlocks(RuntimeError):
    pass


class BlockAllocator:
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))     # a stack: block 0 is handed out first

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, n: int) -> List[int]:
        if n > len(self._free):
            raise OutOfBlocks(f"need {n} blocks, {len(self._free)} free")
        return [self._free.pop() for _ in range(n)]

    def release(self, blocks: List[int]) -> None:
        self._free.extend(blocks)


class PagedKVPool:
    """The block pool for every layer. Shapes are (num_blocks, kv_heads_of_this_shard, block_size, head_dim)."""

    def __init__(self, cfg: ModelConfig, num_blocks: int, block_size: int = 16, tp: TPContext = SINGLE,
                 dtype=torch.float32, device="cpu"):
        shape = (num_blocks, cfg.n_kv_heads // tp.size, block_size, cfg.head_dim)
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(cfg.n_layers)]
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)

    def blocks_for(self, n_tokens: int) -> int:
        return -(-n_tokens // self.block_size)

    def views(self, block_tables: torch.Tensor, positions: torch.Tensor) -> List["PagedLayerView"]:
        """One cache view per layer for a step. block_tables: (B, max_blocks) long, positions: (B, T) long."""
        return [PagedLayerView(self, layer, block_tables, positions) for layer in range(len(self.k))]


class PagedLayerView:
    """What one attention layer sees during one step (the same interface as ContiguousKV.update)."""

    def __init__(self, pool: PagedKVPool, layer: int, block_tables: torch.Tensor, positions: torch.Tensor):
        self.pool, self.layer, self.tables, self.positions = pool, layer, block_tables, positions

    @property
    def length(self) -> int:     # only used when no explicit positions are given; the engine always gives them
        raise NotImplementedError("paged views need explicit positions")

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self.pool.block_size
        b, h, t, d = k.shape
        phys = self.tables.gather(1, self.positions // bs).reshape(-1)       # (B*T,) pool block of each new token
        off = (self.positions % bs).reshape(-1)                                # slot inside that block
        # pool[phys, :, off, :] has shape (B*T, h, d)
        self.pool.k[self.layer][phys, :, off, :] = k.permute(0, 2, 1, 3).reshape(b * t, h, d)
        self.pool.v[self.layer][phys, :, off, :] = v.permute(0, 2, 1, 3).reshape(b * t, h, d)
        # Gather each row's blocks, in table order: (B, nb, h, bs, d) -> (B, h, nb*bs, d), positions 0..nb*bs-1.
        k_all = self.pool.k[self.layer][self.tables].permute(0, 2, 1, 3, 4).reshape(b, h, -1, d)
        v_all = self.pool.v[self.layer][self.tables].permute(0, 2, 1, 3, 4).reshape(b, h, -1, d)
        return k_all, v_all
