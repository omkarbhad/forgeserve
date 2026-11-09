"""Run the same model on 1, 2 and 4 processes and show that the output does not change."""

import torch

from forgeserve import ForgeModel, ModelConfig, TPContext
from forgeserve.tensor_parallel import run_distributed, shard_state_dict


def worker(rank, size, cfg_dict, full_sd, idx):
    cfg = ModelConfig(**cfg_dict)
    tp = TPContext(rank, size)
    model = ForgeModel(cfg, tp).eval()
    model.load_state_dict(shard_state_dict(full_sd, cfg, rank, size))
    logits = model(idx)
    shard_params = sum(p.numel() for n, p in model.named_parameters() if "layers" in n)
    return logits, tp.num_all_reduces, tp.all_reduce_elements, shard_params


if __name__ == "__main__":
    cfg = ModelConfig(vocab_size=256, d_model=128, n_layers=4, n_heads=8, n_kv_heads=4, max_seq_len=64)
    torch.manual_seed(0)
    ref = ForgeModel(cfg).eval()
    idx = torch.randint(0, 256, (2, 16))
    want = ref(idx)
    full_layer_params = sum(p.numel() for p in ref.layers.parameters())
    print(f"single process: {full_layer_params:,} layer parameters")
    for size in (2, 4):
        results = run_distributed(worker, size, cfg.to_dict(), ref.state_dict(), idx)
        err = max((r[0] - want).abs().max().item() for r in results)
        _, calls, elements, shard = results[0]
        print(f"{size} processes: {shard:,} layer parameters each, {calls} all-reduces "
              f"({elements:,} numbers) per forward pass, max logit difference {err:.1e}")
