"""Speculative decoding (Leviathan et al., https://arxiv.org/abs/2211.17192; Chen et al., https://arxiv.org/abs/2302.01318).

A small draft model proposes k tokens one at a time. The large target model checks all k in ONE forward pass,
which costs about as much as generating a single token. Each drafted token x_i is kept with probability
min(1, p_i(x_i) / q_i(x_i)), where p is the target's and q the draft's distribution. At the first rejection, a
replacement is drawn from the normalised residual max(p_i - q_i, 0) and the round stops; if all k are kept, one
bonus token is drawn from the target's next distribution. The result is distributed exactly as if the target
model had generated every token itself, however poor the draft is: the draft only changes the speed.
With temperature 0 both distributions are one-hot, and the output equals greedy decoding of the target.
"""

from typing import List, Tuple

import torch

from .kv import new_contiguous_cache
from .sampling import probs_from_logits


def verify_draft(p: torch.Tensor, q: torch.Tensor, draft_tokens: List[int], generator=None) -> Tuple[List[int], int]:
    """p: (k+1, V) target probabilities, q: (k, V) draft probabilities, draft_tokens: k ints.

    Returns (accepted draft tokens, one extra token from the target's distribution)."""
    accepted: List[int] = []
    for i, x in enumerate(draft_tokens):
        ratio = float(p[i, x] / q[i, x]) if q[i, x] > 0 else 1.0
        if float(torch.rand((), generator=generator)) < min(1.0, ratio):
            accepted.append(x)
            continue
        residual = (p[i] - q[i]).clamp(min=0)
        residual = residual / residual.sum() if residual.sum() > 0 else p[i]
        return accepted, int(torch.multinomial(residual, 1, generator=generator))
    return accepted, int(torch.multinomial(p[len(draft_tokens)], 1, generator=generator))   # the bonus token


@torch.no_grad()
def speculative_generate(target, draft, prompt: List[int], max_new_tokens: int, k: int = 4,
                         temperature: float = 0.0, generator=None):
    """Generate with `target`, drafting k tokens at a time with `draft`. Returns (new tokens, stats).

    Both models must share a vocabulary. One sequence at a time."""
    if target.cfg.vocab_size != draft.cfg.vocab_size:
        raise ValueError("target and draft must share a vocabulary")
    limit = min(target.cfg.max_seq_len, draft.cfg.max_seq_len)
    tokens = list(prompt)
    kv_t, kv_d = new_contiguous_cache(target.cfg.n_layers), new_contiguous_cache(draft.cfg.n_layers)
    if len(tokens) > 1:                                        # caches hold every token except the last
        target(torch.tensor([tokens[:-1]]), kv=kv_t)
        draft(torch.tensor([tokens[:-1]]), kv=kv_d)
    stats = {"rounds": 0, "proposed": 0, "accepted": 0}
    start = len(tokens)

    while len(tokens) - start < max_new_tokens and len(tokens) < limit:
        remaining = min(max_new_tokens - (len(tokens) - start), limit - len(tokens))
        k_eff = min(k, remaining - 1)                          # a round emits at most k_eff + 1 tokens
        cached = kv_t[0].length                                # == len(tokens) - 1

        # 1. the draft model proposes k_eff tokens
        xs, qs, cur = [], [], tokens[-1]
        for _ in range(k_eff):
            q = probs_from_logits(draft(torch.tensor([[cur]]), kv=kv_d)[0, -1], temperature)
            cur = int(q.argmax()) if temperature == 0 else int(torch.multinomial(q, 1, generator=generator))
            xs.append(cur)
            qs.append(q)

        # 2. the target model scores the last token and all proposals in one pass
        p = probs_from_logits(target(torch.tensor([[tokens[-1]] + xs]), kv=kv_t)[0], temperature)   # (k_eff+1, V)

        # 3. keep a prefix of the proposals, plus one token from the target
        accepted, extra = verify_draft(p, torch.stack(qs) if qs else p[:0], xs, generator)
        tokens += accepted + [extra]
        stats["rounds"] += 1
        stats["proposed"] += k_eff
        stats["accepted"] += len(accepted)

        # 4. drop rejected positions from both caches (and feed the draft whatever it has not seen)
        want = cached + 1 + len(accepted)                      # everything except the new last token
        for view in kv_t:
            view.truncate(want)
        have = kv_d[0].length
        if have > want:
            for view in kv_d:
                view.truncate(want)
        elif have < want:
            draft(torch.tensor([tokens[have:want]]), kv=kv_d)
    return tokens[start:], stats
