"""Tensor parallelism (Megatron-LM, https://arxiv.org/abs/1909.08053): split every layer across N processes.

- Column-parallel (q, k, v, gate, up): each shard owns a slice of the output features, which for attention
  means whole heads. No communication is needed after them.
- Row-parallel (attention output, MLP down): each shard owns a slice of the input features and produces a
  partial sum, which one all-reduce adds up. That is 2 all-reduces per layer, 2 * n_layers per forward pass.

`shard_state_dict` cuts a full model's weights into one shard; `run_distributed` starts N processes that talk
over `torch.distributed` (gloo, so it runs on CPU).
"""

import os
import socket
import traceback
from typing import Callable, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import ModelConfig

COLUMN_PARALLEL = (".wq.", ".wk.", ".wv.", ".w_gate.", ".w_up.")   # split the output dimension (rows of the weight)
ROW_PARALLEL = (".wo.", ".w_down.")                                  # split the input dimension (columns)


def shard_state_dict(full: Dict[str, torch.Tensor], cfg: ModelConfig, rank: int, size: int) -> Dict[str, torch.Tensor]:
    """The slice of a full model's weights that shard `rank` of `size` holds."""
    cfg.check_tensor_parallel(size)
    out = {}
    for name, w in full.items():
        if any(tag in name for tag in COLUMN_PARALLEL):
            n = w.shape[0] // size
            out[name] = w[rank * n : (rank + 1) * n].clone().contiguous()
        elif any(tag in name for tag in ROW_PARALLEL):
            n = w.shape[1] // size
            out[name] = w[:, rank * n : (rank + 1) * n].clone().contiguous()
        else:
            out[name] = w.clone()           # norms and embeddings are replicated
    return out


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, size: int, port: int, fn: Callable, args: tuple, queue) -> None:
    torch.set_num_threads(1)                # N processes on one machine: do not oversubscribe the cores
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size)
    try:
        queue.put((rank, fn(rank, size, *args), None))
    except BaseException:
        queue.put((rank, None, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def run_distributed(fn: Callable, size: int, *args, timeout: float = 180.0) -> List:
    """Run fn(rank, size, *args) in `size` processes; return their results ordered by rank.

    `fn` must be a module-level function; args and results must be picklable.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, size, port, fn, args, queue)) for r in range(size)]
    for p in procs:
        p.start()
    results, error = {}, None
    try:
        for _ in range(size):
            rank, res, err = queue.get(timeout=timeout)
            if err is not None:
                error = f"rank {rank} failed:\n{err}"
                break
            results[rank] = res
    except Exception as e:                  # timeout: some rank is stuck, usually in a collective
        error = f"no result within {timeout}s ({type(e).__name__})"
    finally:
        for p in procs:
            if error is not None and p.is_alive():
                p.terminate()
            p.join(timeout=10)
    if error is not None:
        raise RuntimeError(error)
    return [results[r] for r in range(size)]
