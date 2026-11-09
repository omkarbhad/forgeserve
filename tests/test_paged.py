import random

import pytest
import torch

from forgeserve import BlockAllocator, ForgeModel, OutOfBlocks, PagedKVPool, new_contiguous_cache
from conftest import make_model, small_cfg


def test_allocator():
    a = BlockAllocator(4)
    got = a.allocate(3)
    assert sorted(got) == [0, 1, 2] and a.num_free == 1
    with pytest.raises(OutOfBlocks):
        a.allocate(2)
    a.release(got)
    assert a.num_free == 4


def test_write_then_read_back_across_scattered_blocks():
    cfg = small_cfg()
    pool = PagedKVPool(cfg, num_blocks=10, block_size=4)
    table = torch.tensor([[7, 2, 9]])                 # physical blocks, deliberately not in order
    k = torch.randn(1, cfg.n_kv_heads, 10, cfg.head_dim)
    v = torch.randn_like(k)
    positions = torch.arange(10)[None]
    k_all, v_all = pool.views(table, positions)[0].update(k, v)
    assert k_all.shape == (1, cfg.n_kv_heads, 12, cfg.head_dim)     # 3 blocks * 4 slots
    torch.testing.assert_close(k_all[:, :, :10], k)
    torch.testing.assert_close(v_all[:, :, :10], v)
    # the data really lives in those physical blocks, at the right slots
    torch.testing.assert_close(pool.k[0][2, :, 1], k[0, :, 5])       # position 5 -> block 1 of the table -> pool block 2, slot 1


def test_two_rows_do_not_overwrite_each_other():
    cfg = small_cfg()
    pool = PagedKVPool(cfg, num_blocks=8, block_size=4)
    tables = torch.tensor([[0, 1], [2, 3]])
    k = torch.randn(2, cfg.n_kv_heads, 6, cfg.head_dim)
    k_all, _ = pool.views(tables, torch.arange(6)[None].expand(2, 6))[0].update(k, k.clone())
    torch.testing.assert_close(k_all[:, :, :6], k)


def test_paged_forward_matches_contiguous_forward():
    """Prefill plus token-by-token decoding through fragmented blocks gives the same logits as a plain cache."""
    cfg = small_cfg()
    model = make_model(cfg)
    idx = torch.randint(0, 64, (1, 30))
    pool = PagedKVPool(cfg, num_blocks=16, block_size=4)
    blocks = pool.allocator.allocate(8)
    random.Random(0).shuffle(blocks)                  # fragmentation
    table = torch.tensor([blocks])

    kv = new_contiguous_cache(cfg.n_layers)
    want = [model(idx[:, :12], kv=kv)]
    got = [model(idx[:, :12], torch.arange(12)[None], pool.views(table, torch.arange(12)[None]))]
    for i in range(12, 30):
        want.append(model(idx[:, i : i + 1], kv=kv))
        pos = torch.tensor([[i]])
        got.append(model(idx[:, i : i + 1], pos, pool.views(table, pos)))
    torch.testing.assert_close(torch.cat(got, 1), torch.cat(want, 1), atol=1e-4, rtol=1e-4)
