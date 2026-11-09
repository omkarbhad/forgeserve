import pytest
import torch

from forgeserve import ForgeModel, ModelConfig, TPContext, generate, new_contiguous_cache
from forgeserve.tensor_parallel import run_distributed, shard_state_dict
from conftest import small_cfg


def forward_worker(rank, size, cfg_dict, full_sd, idx):
    """One shard: load its slice of the weights, run a forward pass and a cached greedy generation."""
    cfg = ModelConfig(**cfg_dict)
    tp = TPContext(rank, size)
    model = ForgeModel(cfg, tp).eval()
    model.load_state_dict(shard_state_dict(full_sd, cfg, rank, size))
    logits = model(idx)
    forward_calls = tp.num_all_reduces, tp.all_reduce_elements
    gen = generate(model, idx[:, :4], 8)
    n_params = sum(p.numel() for n, p in model.named_parameters() if "layers" in n)
    return logits, forward_calls, gen, n_params


@pytest.mark.parametrize("size", [2, 4])
def test_tensor_parallel_matches_single_device(size):
    cfg = small_cfg()
    ref = ForgeModel(cfg).eval()
    idx = torch.randint(0, 64, (3, 12))
    results = run_distributed(forward_worker, size, cfg.to_dict(), ref.state_dict(), idx)

    want_logits = ref(idx)
    want_gen = generate(ref, idx[:, :4], 8)
    layer_params = sum(p.numel() for n, p in ref.named_parameters() if "layers" in n)
    norm_params = sum(p.numel() for n, p in ref.named_parameters() if "layers" in n and "norm" in n)

    for rank, (logits, (calls, elements), gen, n_params) in enumerate(results):
        torch.testing.assert_close(logits, want_logits, atol=1e-4, rtol=1e-4)
        assert torch.equal(gen, want_gen)                       # greedy tokens identical, with the KV cache
        # exactly two all-reduces per layer, each over a (batch, seq, d_model) tensor
        assert calls == 2 * cfg.n_layers
        assert elements == 2 * cfg.n_layers * 3 * 12 * cfg.d_model
        # weights are divided across shards; only the norms are replicated
        assert n_params == (layer_params - norm_params) // size + norm_params


def test_sharded_weights_rebuild_the_full_model():
    cfg = small_cfg()
    full = ForgeModel(cfg).state_dict()
    shards = [shard_state_dict(full, cfg, r, 2) for r in range(2)]
    for name, w in full.items():
        if any(t in name for t in (".wq.", ".wk.", ".wv.", ".w_gate.", ".w_up.")):
            torch.testing.assert_close(torch.cat([s[name] for s in shards], dim=0), w)
        elif any(t in name for t in (".wo.", ".w_down.")):
            torch.testing.assert_close(torch.cat([s[name] for s in shards], dim=1), w)
        else:
            torch.testing.assert_close(shards[0][name], w)


def test_indivisible_sizes_are_rejected():
    with pytest.raises(ValueError):
        ForgeModel(small_cfg(), TPContext(0, 3))
    with pytest.raises(ValueError):
        shard_state_dict(ForgeModel(small_cfg()).state_dict(), small_cfg(), 0, 8)   # n_kv_heads = 4


def test_failing_worker_is_reported_not_hung():
    with pytest.raises(RuntimeError, match="rank"):
        run_distributed(crash_worker, 2, timeout=60)


def crash_worker(rank, size):
    raise ValueError("boom")
