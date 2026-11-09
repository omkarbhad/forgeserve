# forgeserve

A small LLM inference engine written from scratch in PyTorch (about 840 lines), built around the four ideas that
large-scale inference rests on:

| Piece | What it does | Paper |
| --- | --- | --- |
| **Tensor parallelism** | splits every layer across N processes, 2 all-reduces per layer | [Megatron-LM](https://arxiv.org/abs/1909.08053) |
| **Pipeline parallelism** | splits the layers into stages; micro-batches flow through them | [GPipe](https://arxiv.org/abs/1811.06965) |
| **Paged KV cache + continuous batching** | a block pool instead of one big cache per request; requests join and leave the batch at every step | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) |
| **Speculative decoding** | a small model drafts, the big model verifies many tokens in one pass, same output distribution | [Leviathan et al.](https://arxiv.org/abs/2211.17192) |

The model underneath is a LLaMA-style decoder (RMSNorm, grouped-query attention, rotary embeddings, SwiGLU).
It is written once and used by all four pieces.

## What this is, and what it is not

**This is a correctness-first implementation.** Every piece is checked against the plain single-process model:
the logits and the generated tokens must be the same, and the tests fail when I break the code on purpose.

**It is not a fast engine, and I have not measured speed.** I tested only on a laptop CPU, with `gloo` between
processes on one machine. I did not run real GPUs, NCCL, or more than one machine, and there are no custom
kernels (the paged attention gathers blocks with ordinary PyTorch indexing). Any timing on this setup would say
nothing about a real cluster, so the README reports none. What the repo does show is the data flow of each
technique, with numbers that do not depend on the hardware: how many collectives run, how many numbers they move,
how much cache memory is used, and how many tokens are accepted.

## Install

```bash
git clone https://github.com/omkarbhad/forgeserve
cd forgeserve
pip install -e ".[dev]"
pytest -q          # 46 tests, about 25 seconds, includes multi-process runs
```

## 1. Tensor parallelism

Each layer's weights are cut across N processes. The q, k, v, gate and up projections are **column-parallel**:
a shard owns a slice of the output features (for attention, whole heads), so no communication is needed after
them. The attention output and MLP down projections are **row-parallel**: a shard owns a slice of the input
features and produces a partial sum, and one all-reduce adds the partial sums up.

```python
from forgeserve import ForgeModel, ModelConfig, TPContext
from forgeserve.tensor_parallel import run_distributed, shard_state_dict

def worker(rank, size, cfg_dict, full_state_dict, idx):
    cfg = ModelConfig(**cfg_dict)
    model = ForgeModel(cfg, TPContext(rank, size)).eval()
    model.load_state_dict(shard_state_dict(full_state_dict, cfg, rank, size))
    return model(idx)

results = run_distributed(worker, 4, cfg.to_dict(), full.state_dict(), idx)   # 4 processes, gloo on CPU
```

`python examples/tensor_parallel_demo.py` on a 4-layer model prints:

```
single process: 787,456 layer parameters
2 processes: 394,240 layer parameters each, 8 all-reduces (32,768 numbers) per forward pass, max logit difference 3.6e-07
4 processes: 197,632 layer parameters each, 8 all-reduces (32,768 numbers) per forward pass, max logit difference 4.2e-07
```

Weights divide across shards (only the small norm weights are copied), and the output matches to float rounding.
The number of all-reduces is `2 * n_layers`, whatever the number of shards. A test checks the count and the
exact element count on every rank.

Not implemented: a vocabulary-parallel embedding and output head (they are replicated on every shard), and
tensor parallelism with more shards than KV heads.

## 2. Paged KV cache and continuous batching

A normal cache reserves `max_seq_len` slots for every request, even if most are never used. Here the cache is one
pool of fixed-size blocks. A request owns a block table (a list of blocks that need not be next to each other in
memory) and takes a new block only when its last one is full, so memory is wasted only in the last block.

The engine takes requests at any time. Every step it either prefills one waiting request, or decodes one token
for **all** running requests in a single batched forward pass, so a finished request frees its place at once
instead of waiting for the whole batch. If the pool runs out, the most recent request is preempted: its blocks are
freed, and it is re-prefilled later from its prompt plus the tokens it has made (recompute preemption, as in vLLM).

```python
from forgeserve import Engine

engine = Engine(model, num_blocks=64, block_size=16, max_batch=8)
a = engine.add_request([1, 2, 3], max_new_tokens=20)
b = engine.add_request([4, 5], max_new_tokens=5, temperature=0.8)
outputs = engine.run()       # {request id: generated tokens}
```

`python examples/serve_demo.py` serves 12 requests of different lengths through a 24-block pool:

```
requests: 12   engine steps: 66   preemptions: 0
largest batch: 6 sequences   peak pool use: 22/24 blocks = 352 token slots
reserving max_seq_len for that batch would take 1536 slots (4.4x more)
every output identical to one-sequence-at-a-time generation: True
```

The engine runs unchanged under tensor parallelism: every shard runs the same loop and its pool holds only its own
KV heads (a test runs it on 2 processes with a pool small enough to force preemption).

Not implemented: sharing blocks between requests (prefix caching), chunked prefill (a prefill step does not mix with
decoding), and a fused attention kernel.

## 3. Speculative decoding

A small draft model proposes `k` tokens, one at a time. The target model scores all of them in one forward pass. A
proposed token `x` is kept with probability `min(1, p(x) / q(x))`, where `p` is the target's and `q` the draft's
probability. At the first rejection, a replacement is drawn from the normalised `max(p - q, 0)` and the round
stops. If all `k` are kept, one extra token comes from the target. **The output is distributed exactly as if the
target had generated alone, however bad the draft is; the draft only changes how many target passes are needed.**
With temperature 0 the output equals greedy decoding of the target.

```python
from forgeserve import speculative_generate

tokens, stats = speculative_generate(target, draft, prompt, max_new_tokens=64, k=4, temperature=0.0)
stats   # {'rounds': ..., 'proposed': ..., 'accepted': ...}
```

Both models keep a KV cache that is cut back after every round to drop the rejected positions. One sequence at a
time; a tree of drafts and batched speculation are not implemented.

## 4. Pipeline parallelism

The layers are split into consecutive stages, one process each. Stage 0 holds the embeddings, the last stage holds
the output head, and each stage holds its own slice of the KV cache. The batch is cut into micro-batches that move
from stage to stage with `send`/`recv`, so stage 1 can work on micro-batch 0 while stage 0 starts micro-batch 1.
With `p` stages and `m` micro-batches a share `(p - 1) / (m + p - 1)` of stage time is idle (`bubble_fraction`).

```python
from forgeserve import PipelineStage, pipeline_generate

stage = PipelineStage(cfg, rank, size)
stage.load_from_full(full_state_dict)
tokens = pipeline_generate(stage, prompt, max_new_tokens=32, microbatches=2)   # same result on every stage
```

Not implemented: combining tensor and pipeline parallelism in one run, and overlapping decoding steps of
different batches.

## How it is tested

46 tests. The main checks:

- **Tensor parallelism:** logits from 2 and 4 processes equal the single-process logits, cached greedy generation
  gives the same tokens, there are exactly `2 * n_layers` all-reduces, and the weights are divided as expected.
  A crashing worker is reported instead of hanging.
- **Paged cache:** writing and reading back across scattered blocks returns exactly what was written, and a
  prefill plus token-by-token decode through shuffled blocks matches a plain cache.
- **Engine:** every request's output equals one-at-a-time generation, for batched requests, joining and leaving the
  batch, forced preemption, end-of-sequence and length limits. All blocks are returned at the end. A guard test checks that the test model produces varied tokens.
- **Speculative decoding:** greedy output equals the target's for several draft lengths and an unrelated draft; a
  perfect draft is always accepted; the verification step reproduces the target distribution over 60,000 trials;
  through the real models on a tiny vocabulary the first token follows the target's own distribution.
- **Pipeline:** logits from 2 and 3 stages, with 1 to 4 micro-batches, equal the single-process logits, including a
  second chunk that continues from the stage caches; generation matches on every stage.

I broke the code on purpose to confirm the tests notice. Examples: skip an all-reduce, shard the wrong axis, shift a
slot offset, drop the re-prefill after preemption, invert the acceptance ratio, resample from `p` instead of the
residual, forget to cut the cache after a rejection, mix up the micro-batch caches. One of these first slipped
through, because a randomly initialised model repeats a single token forever and hides lost tokens. The tests now use a
model with larger weights, and a guard test keeps it that way.

## Layout

```
forgeserve/
  config.py           ModelConfig
  model.py            the transformer: explicit positions, cache views, built-in tensor parallelism
  parallel.py         TPContext (rank, size, all-reduce with counters)
  tensor_parallel.py  shard_state_dict, run_distributed (spawns gloo processes)
  kv.py               plain KV cache
  paged.py            block allocator, block pool, per-layer paged views
  engine.py           continuous batching, preemption
  speculative.py      draft, verify, cache rollback
  pipeline.py         stages, micro-batches, send/recv
tests/                46 tests, includes multi-process runs
examples/             serve_demo.py, tensor_parallel_demo.py
```

## Citations

```bibtex
@article{shoeybi2019megatron,
    title   = {Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
    author  = {Mohammad Shoeybi and Mostofa Patwary and Raul Puri and Patrick LeGresley and Jared Casper and Bryan Catanzaro},
    journal = {arXiv preprint arXiv:1909.08053},
    year    = {2019}
}
@inproceedings{kwon2023vllm,
    title     = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
    author    = {Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
    booktitle = {Symposium on Operating Systems Principles},
    year      = {2023}
}
@inproceedings{leviathan2023speculative,
    title     = {Fast Inference from Transformers via Speculative Decoding},
    author    = {Yaniv Leviathan and Matan Kalman and Yossi Matias},
    booktitle = {International Conference on Machine Learning},
    year      = {2023}
}
@inproceedings{huang2019gpipe,
    title     = {GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism},
    author    = {Yanping Huang and Youlong Cheng and Ankur Bapna and Orhan Firat and Mia Xu Chen and Dehao Chen and HyoukJoong Lee and Jiquan Ngiam and Quoc V. Le and Yonghui Wu and Zhifeng Chen},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2019}
}
```

## License

MIT
