"""Continuous batching over a paged KV cache: 12 requests of different lengths share one small block pool."""

import torch

from forgeserve import Engine, ForgeModel, ModelConfig, generate

cfg = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=8, n_kv_heads=4, max_seq_len=256)
torch.manual_seed(0)
model = ForgeModel(cfg).eval()
for p in model.parameters():                 # random weights; make them big enough for varied tokens
    if p.ndim >= 2:
        p.data = torch.randn_like(p) * 0.25

g = torch.Generator().manual_seed(1)
requests = [(torch.randint(0, 256, (int(torch.randint(5, 60, (1,), generator=g)),), generator=g).tolist(),
             int(torch.randint(8, 40, (1,), generator=g))) for _ in range(12)]

block_size, num_blocks = 16, 24
engine = Engine(model, num_blocks=num_blocks, block_size=block_size, max_batch=6)
for prompt, n in requests:
    engine.add_request(prompt, n)
out = engine.run()

ok = all(out[i] == generate(model, torch.tensor([p]), n)[0].tolist()[len(p):] for i, (p, n) in enumerate(requests))
reserved = engine.peak_running * cfg.max_seq_len
print(f"requests: {len(requests)}   engine steps: {engine.steps}   preemptions: {engine.preemptions}")
print(f"largest batch: {engine.peak_running} sequences   peak pool use: {engine.peak_blocks}/{num_blocks} blocks "
      f"= {engine.peak_blocks * block_size} token slots")
print(f"reserving max_seq_len for that batch would take {reserved} slots ({reserved / (engine.peak_blocks * block_size):.1f}x more)")
print("every output identical to one-sequence-at-a-time generation:", ok)
