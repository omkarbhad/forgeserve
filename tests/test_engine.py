import pytest
import torch

from forgeserve import Engine, ForgeModel, generate
from conftest import make_model, small_cfg


def reference(model, prompt, n, eos_id=None):
    out = generate(model, torch.tensor([prompt]), n)[0].tolist()[len(prompt):]
    if eos_id is not None and eos_id in out:
        out = out[: out.index(eos_id) + 1]
    return out


def make_requests(n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i in range(n):
        plen = int(torch.randint(2, 20, (1,), generator=g))
        reqs.append((torch.randint(0, 64, (plen,), generator=g).tolist(), int(torch.randint(3, 15, (1,), generator=g))))
    return reqs


def test_every_request_matches_the_reference_generator(model):
    reqs = make_requests()
    engine = Engine(model, num_blocks=64, block_size=4, max_batch=3)
    ids = [engine.add_request(p, n) for p, n in reqs]
    out = engine.run()
    for rid, (p, n) in zip(ids, reqs):
        assert out[rid] == reference(model, p, n), rid
    assert engine.peak_running > 1                    # sequences really were batched together
    assert engine.pool.allocator.num_free == 64       # all blocks returned
    assert engine.preemptions == 0


def test_sequences_join_and_leave_the_batch():
    """A short request finishes while a long one is still running, and a waiting one takes its place."""
    model = make_model()
    engine = Engine(model, num_blocks=64, block_size=4, max_batch=2)
    a = engine.add_request([1, 2, 3], 12)
    b = engine.add_request([4, 5], 2)
    c = engine.add_request([6, 7, 8, 9], 4)
    seen = []
    while engine.has_work():
        engine.step()
        seen.append(tuple(sorted(r.id for r in engine.running)))
    assert (a, b) in seen and (a, c) in seen          # c entered after b left, while a kept running
    assert engine.requests[a].output == reference(model, [1, 2, 3], 12)
    assert engine.requests[c].output == reference(model, [6, 7, 8, 9], 4)


def test_preemption_keeps_outputs_correct():
    model = make_model()
    reqs = make_requests(5)
    engine = Engine(model, num_blocks=12, block_size=4, max_batch=5)   # far too small for all at once
    ids = [engine.add_request(p, n) for p, n in reqs]
    out = engine.run()
    assert engine.preemptions > 0
    for rid, (p, n) in zip(ids, reqs):
        assert out[rid] == reference(model, p, n), rid
    assert engine.pool.allocator.num_free == 12


def test_blocks_are_allocated_on_demand():
    model = make_model()
    engine = Engine(model, num_blocks=64, block_size=4, max_batch=4)
    for p, n in make_requests(4):
        engine.add_request(p, n)
    while engine.has_work():
        engine.step()
        for r in engine.running:
            assert len(r.blocks) == -(-r.cached // 4) or len(r.blocks) == r.cached // 4 + 1   # no more than one spare block
    # paged memory is far below reserving max_seq_len for each of the batch's sequences
    assert engine.peak_blocks * 4 < engine.peak_running * model.cfg.max_seq_len


def test_stops_at_eos():
    model = make_model()
    prompt = [3, 1, 4, 1, 5]
    full = reference(model, prompt, 10)
    eos = full[3]
    engine = Engine(model, num_blocks=32, block_size=4)
    rid = engine.add_request(prompt, 10, eos_id=eos)
    assert engine.run()[rid] == full[: full.index(eos) + 1]


def test_stops_at_max_seq_len():
    model = make_model()
    engine = Engine(model, num_blocks=64, block_size=4)
    rid = engine.add_request(list(range(1, 60)), 100)
    out = engine.run()[rid]
    assert 59 + len(out) == model.cfg.max_seq_len


def test_temperature_sampling_is_seeded_and_valid():
    model = make_model()
    def run(seed):
        e = Engine(model, num_blocks=32, block_size=4, seed=seed)
        rid = e.add_request([1, 2, 3], 10, temperature=1.0)
        return e.run()[rid]
    a, b, c = run(0), run(0), run(1)
    assert a == b and len(a) == 10 and all(0 <= t < 64 for t in a)
    assert a != c


def test_invalid_requests():
    model = make_model()
    engine = Engine(model, num_blocks=4, block_size=4)
    with pytest.raises(ValueError):
        engine.add_request([], 5)
    with pytest.raises(ValueError):
        engine.add_request(list(range(1, 40)), 5)       # needs more blocks than the pool has


def test_pool_too_small_for_one_sequence_raises_instead_of_looping():
    model = make_model()
    engine = Engine(model, num_blocks=2, block_size=4)
    engine.add_request([1, 2, 3, 4, 5], 30)              # grows beyond 8 slots
    with pytest.raises(RuntimeError):
        engine.run()


def test_the_test_model_generates_varied_tokens():
    """Guards the other tests: a model that repeats one token forever would hide lost or duplicated tokens."""
    model = make_model()
    distinct = [len(set(reference(model, p, 20))) for p, _ in make_requests(6)]
    assert sum(distinct) / len(distinct) >= 6, distinct
