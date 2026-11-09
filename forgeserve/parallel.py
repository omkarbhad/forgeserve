"""Tensor-parallel context: which shard this process is, and how to sum partial results across shards."""

import torch
import torch.distributed as dist


class TPContext:
    """`size` shards; this process is shard `rank`. With size 1 every operation is a no-op.

    Inference only: all_reduce here is not autograd-aware.
    """

    def __init__(self, rank: int = 0, size: int = 1, group=None):
        if not 0 <= rank < size:
            raise ValueError("rank must be in [0, size)")
        self.rank, self.size, self.group = rank, size, group
        self.num_all_reduces = 0      # how many collectives this process has run
        self.all_reduce_elements = 0  # and how many numbers they moved

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum `t` over all shards, in place, and return it."""
        if self.size > 1:
            self.num_all_reduces += 1
            self.all_reduce_elements += t.numel()
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.group)
        return t


SINGLE = TPContext()
