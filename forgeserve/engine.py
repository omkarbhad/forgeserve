"""A small serving engine: continuous batching over a paged KV cache.

Requests arrive at any time. Each engine step either prefills one waiting request (a whole prompt in one
forward pass) or decodes one token for every running request in one batched forward pass, so sequences join
and leave the batch whenever they start or finish, instead of waiting for a whole batch to end.

If the block pool runs out during decoding, the most recently started request is preempted: its blocks are
freed and it goes back to the front of the queue, to be re-prefilled later from its prompt plus the tokens it
has produced so far (recompute preemption, as in vLLM).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import torch

from .model import ForgeModel
from .paged import PagedKVPool
from .sampling import sample


@dataclass
class Request:
    id: int
    prompt: List[int]
    max_new_tokens: int
    eos_id: Optional[int] = None
    temperature: float = 0.0
    output: List[int] = field(default_factory=list)
    blocks: List[int] = field(default_factory=list)    # block table while the request holds cache blocks
    cached: int = 0                                     # tokens whose keys/values are in the cache
    finished: bool = False


class Engine:
    def __init__(self, model: ForgeModel, num_blocks: int, block_size: int = 16, max_batch: int = 8, seed: int = 0):
        self.model, self.cfg = model, model.cfg
        self.pool = PagedKVPool(self.cfg, num_blocks, block_size, tp=model.tp)
        self.max_batch = max_batch
        self.generator = torch.Generator().manual_seed(seed)
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.requests: Dict[int, Request] = {}
        self.steps = self.preemptions = self.peak_running = self.peak_blocks = 0

    # ---- public API -------------------------------------------------------------------------------------
    def add_request(self, prompt: List[int], max_new_tokens: int, eos_id: Optional[int] = None,
                    temperature: float = 0.0) -> int:
        if not 0 < len(prompt) < self.cfg.max_seq_len:
            raise ValueError(f"prompt length must be in 1..{self.cfg.max_seq_len - 1}")
        if self.pool.blocks_for(len(prompt) + 1) > self.pool.allocator.num_blocks:
            raise ValueError("prompt does not fit in the block pool")
        req = Request(len(self.requests), list(prompt), max_new_tokens, eos_id, temperature)
        self.requests[req.id] = req
        self.waiting.append(req)
        return req.id

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def run(self) -> Dict[int, List[int]]:
        """Step until every request is finished. Returns the generated tokens of each request."""
        while self.has_work():
            self.step()
        return {rid: r.output for rid, r in self.requests.items()}

    def step(self) -> None:
        self.steps += 1
        if self.waiting and len(self.running) < self.max_batch and self._try_prefill():
            pass
        elif self.running:
            self._decode()
        elif self.waiting:
            raise RuntimeError("the head request cannot be scheduled: the block pool is too small")
        used = self.pool.allocator.num_blocks - self.pool.allocator.num_free
        self.peak_running, self.peak_blocks = max(self.peak_running, len(self.running)), max(self.peak_blocks, used)

    # ---- scheduling -------------------------------------------------------------------------------------
    def _try_prefill(self) -> bool:
        req = self.waiting[0]
        tokens = req.prompt + req.output                       # after a preemption this includes earlier output
        need = self.pool.blocks_for(len(tokens))
        if need > self.pool.allocator.num_free:
            return False
        self.waiting.popleft()
        req.blocks = self.pool.allocator.allocate(need)
        positions = torch.arange(len(tokens))[None]
        views = self.pool.views(torch.tensor([req.blocks]), positions)
        logits = self.model(torch.tensor([tokens]), positions, views)
        req.cached = len(tokens)
        self._append(req, logits[0, -1])
        if not req.finished:
            self.running.append(req)
        return True

    def _decode(self) -> None:
        bs = self.pool.block_size
        # every running request writes one new token at position `cached`; some need a fresh block for it
        i = 0
        while i < len(self.running):
            req = self.running[i]
            if req.cached % bs == 0 and req.cached // bs == len(req.blocks):
                if self.pool.allocator.num_free == 0:
                    if len(self.running) == 1:                 # preempting it would only make it start over
                        raise RuntimeError("the block pool is too small to hold even one sequence of this length")
                    victim = self.running.pop()                # the most recently started request
                    self.pool.allocator.release(victim.blocks)
                    victim.blocks, victim.cached = [], 0
                    self.waiting.appendleft(victim)
                    self.preemptions += 1
                    if victim is req:
                        continue                               # it was the one waiting for a block
                    if not self.running:
                        return
                    continue                                   # retry this request: a block is free now
                req.blocks += self.pool.allocator.allocate(1)
            i += 1
        if not self.running:
            return
        max_blocks = max(len(r.blocks) for r in self.running)
        tables = torch.tensor([r.blocks + [0] * (max_blocks - len(r.blocks)) for r in self.running])
        positions = torch.tensor([[r.cached] for r in self.running])
        idx = torch.tensor([[r.output[-1]] for r in self.running])
        logits = self.model(idx, positions, self.pool.views(tables, positions))
        for row, req in enumerate(list(self.running)):
            req.cached += 1
            self._append(req, logits[row, -1])
            if req.finished:
                self.running.remove(req)

    def _append(self, req: Request, logits: torch.Tensor) -> None:
        token = int(sample(logits, req.temperature, self.generator))
        req.output.append(token)
        total = len(req.prompt) + len(req.output)
        if (len(req.output) >= req.max_new_tokens or token == req.eos_id or total >= self.cfg.max_seq_len):
            req.finished = True
            self.pool.allocator.release(req.blocks)
            req.blocks = []
