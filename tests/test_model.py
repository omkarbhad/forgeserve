import pytest
import torch

from forgeserve import ForgeModel, ModelConfig, generate, new_contiguous_cache
from conftest import small_cfg


def test_config_validation():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(n_heads=8, n_kv_heads=3)
    with pytest.raises(ValueError):
        small_cfg().check_tensor_parallel(3)
    small_cfg().check_tensor_parallel(4)


def test_shapes(model, cfg):
    assert model(torch.randint(0, 64, (3, 10))).shape == (3, 10, 64)


def test_no_future_leakage(model):
    a = torch.randint(0, 64, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 64, (4,))
    la, lb = model(a), model(b)
    torch.testing.assert_close(la[:, :8], lb[:, :8], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:])


def test_cache_one_token_at_a_time_matches_full_forward(model, cfg):
    idx = torch.randint(0, 64, (2, 20))
    kv = new_contiguous_cache(cfg.n_layers)
    steps = torch.cat([model(idx[:, i : i + 1], kv=kv) for i in range(20)], dim=1)
    torch.testing.assert_close(steps, model(idx), atol=1e-4, rtol=1e-4)


def test_chunk_after_cached_prefix(model, cfg):
    idx = torch.randint(0, 64, (2, 24))
    kv = new_contiguous_cache(cfg.n_layers)
    model(idx[:, :10], kv=kv)
    torch.testing.assert_close(model(idx[:, 10:], kv=kv), model(idx)[:, 10:], atol=1e-4, rtol=1e-4)


def test_rows_at_different_positions_match_separate_runs(model):
    """Continuous batching puts sequences of different lengths in one batch: positions are per row."""
    a, b = torch.randint(0, 64, (1, 7)), torch.randint(0, 64, (1, 3))
    want_a, want_b = model(a)[:, -1], model(b)[:, -1]
    # one step with the last token of each row, at its own position (keys of the earlier tokens come from the cache)
    kv_a, kv_b = new_contiguous_cache(3), new_contiguous_cache(3)
    model(a[:, :-1], kv=kv_a)
    model(b[:, :-1], kv=kv_b)
    got_a = model(a[:, -1:], kv=kv_a)[:, -1]
    got_b = model(b[:, -1:], kv=kv_b)[:, -1]
    torch.testing.assert_close(got_a, want_a, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(got_b, want_b, atol=1e-4, rtol=1e-4)


def test_explicit_positions_are_used(model):
    idx = torch.randint(0, 64, (1, 5))
    base = model(idx)
    shifted = model(idx, positions=torch.arange(5)[None] + 11)
    assert not torch.allclose(base, shifted)            # rotary positions change the output
    again = model(idx, positions=torch.arange(5)[None])
    torch.testing.assert_close(base, again)


def test_greedy_generate_matches_naive_loop(model):
    prompt = torch.randint(0, 64, (2, 5))
    got = generate(model, prompt, 10)
    idx = prompt
    for _ in range(10):
        idx = torch.cat([idx, model(idx)[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(got, idx)


def test_length_limit(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 65)))
    assert generate(model, torch.randint(0, 64, (1, 60)), 100).shape[1] == 64
