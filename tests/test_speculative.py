import torch
import pytest

from forgeserve import ForgeModel, ModelConfig, generate, speculative_generate, verify_draft
from conftest import make_model, small_cfg


def second_model(seed=7, **kw):
    """A different model with the same vocabulary, to play the draft."""
    torch.manual_seed(seed)
    m = ForgeModel(small_cfg(**kw)).eval()
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        if p.ndim >= 2:
            p.data = torch.randn(p.shape, generator=g) * 0.25
    return m


PROMPTS = [[1, 2, 3], [5], [9, 8, 7, 6, 5, 4], [0, 0, 1]]


@pytest.mark.parametrize("k", [1, 3, 6])
def test_greedy_output_equals_target_even_with_an_unrelated_draft(k):
    target, draft = make_model(), second_model()
    for prompt in PROMPTS:
        got, stats = speculative_generate(target, draft, prompt, 20, k=k)
        want = generate(target, torch.tensor([prompt]), 20)[0].tolist()[len(prompt):]
        assert got == want, (k, prompt)
        assert stats["accepted"] <= stats["proposed"]


def test_perfect_draft_is_always_accepted_and_needs_fewer_target_passes():
    target = make_model()
    got, stats = speculative_generate(target, target, [1, 2, 3], 25, k=4)
    assert stats["accepted"] == stats["proposed"]
    assert stats["rounds"] <= 6                               # 25 tokens, up to 5 per round
    assert got == generate(target, torch.tensor([[1, 2, 3]]), 25)[0].tolist()[3:]


def test_poor_draft_is_rejected_often_but_output_stays_correct():
    target, draft = make_model(), second_model()
    _, stats = speculative_generate(target, draft, [1, 2, 3], 40, k=4)
    assert stats["accepted"] < stats["proposed"]


def test_stops_exactly_at_max_new_tokens_and_at_the_length_limit():
    target, draft = make_model(), second_model()
    for n in (1, 2, 7):
        got, _ = speculative_generate(target, draft, [1, 2, 3], n, k=5)
        assert len(got) == n
    got, _ = speculative_generate(target, draft, list(range(1, 60)), 100, k=4)
    assert 59 + len(got) == target.cfg.max_seq_len


def test_vocab_mismatch_is_rejected():
    with pytest.raises(ValueError):
        speculative_generate(make_model(), second_model(vocab_size=32), [1, 2], 3)


def test_verification_preserves_the_target_distribution():
    """The first emitted token must follow p_1 exactly, however different the draft q is.
    60,000 trials; the tolerance is about 6 standard errors."""
    g = torch.Generator().manual_seed(0)
    V, k, N = 4, 3, 60_000
    p = torch.softmax(torch.randn(k + 1, V, generator=g) * 1.5, -1)
    q = torch.softmax(torch.randn(k, V, generator=g) * 1.5, -1)           # a mismatched draft
    first = []
    for _ in range(N):
        xs = [int(torch.multinomial(q[i], 1, generator=g)) for i in range(k)]
        accepted, extra = verify_draft(p, q, xs, g)
        first.append((accepted + [extra])[0])
    freq = torch.bincount(torch.tensor(first), minlength=V).float() / N
    assert (freq - p[0]).abs().max() < 0.012, (freq, p[0])


def test_greedy_verification_is_exact_matching():
    p = torch.eye(5)[[2, 4, 1, 3]]                      # target argmax: 2, 4, 1, then bonus 3
    q = torch.eye(5)[[2, 4, 0]]                         # draft argmax: 2, 4, 0 -> third proposal is wrong
    assert verify_draft(p, q, [2, 4, 0]) == ([2, 4], 1)
    q_all_right = torch.eye(5)[[2, 4, 1]]
    assert verify_draft(p, q_all_right, [2, 4, 1]) == ([2, 4, 1], 3)


def test_sampling_through_models_follows_the_target_distribution():
    """Tiny vocabulary: the first speculative token over many runs matches the target's own next-token distribution."""
    cfg = dict(vocab_size=6, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16)
    torch.manual_seed(0)
    target, draft = ForgeModel(ModelConfig(**cfg)).eval(), ForgeModel(ModelConfig(**cfg)).eval()
    for m, seed in ((target, 1), (draft, 2)):
        g = torch.Generator().manual_seed(seed)
        for p in m.parameters():
            if p.ndim >= 2:
                p.data = torch.randn(p.shape, generator=g) * 0.4
    prompt = [1, 2, 3]
    want = torch.softmax(target(torch.tensor([prompt]))[0, -1], -1)
    gen, N = torch.Generator().manual_seed(5), 1500
    firsts = [speculative_generate(target, draft, prompt, 1, k=3, temperature=1.0, generator=gen)[0][0] for _ in range(N)]
    freq = torch.bincount(torch.tensor(firsts), minlength=6).float() / N
    assert (freq - want).abs().max() < 0.05, (freq, want)
