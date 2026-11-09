import pytest
import torch

from forgeserve import (ModelConfig, PipelineStage, PipelineState, bubble_fraction, generate, pipeline_forward,
                        pipeline_generate, stage_layers)
from forgeserve.tensor_parallel import run_distributed
from conftest import make_model, small_cfg


def test_stage_layers_cover_every_layer_once():
    assert stage_layers(3, 3) == [(0, 1), (1, 2), (2, 3)]
    assert stage_layers(7, 3) == [(0, 3), (3, 5), (5, 7)]
    with pytest.raises(ValueError):
        stage_layers(2, 3)


def test_bubble_fraction():
    assert bubble_fraction(1, 5) == 0
    assert bubble_fraction(4, 1) == 0.75               # one micro-batch: three of four stages idle at any time
    assert bubble_fraction(4, 13) == pytest.approx(3 / 16)
    assert bubble_fraction(4, 32) < bubble_fraction(4, 8)


def forward_worker(rank, size, cfg_dict, full_sd, idx, microbatches):
    cfg = ModelConfig(**cfg_dict)
    stage = PipelineStage(cfg, rank, size).eval()
    stage.load_from_full(full_sd)
    state = PipelineState(stage, idx.shape[0], microbatches)
    logits = pipeline_forward(stage, state, idx if rank == 0 else None, idx.shape[1])
    # a second chunk, continuing from the per-stage caches
    more = torch.randint(0, cfg.vocab_size, (idx.shape[0], 3), generator=torch.Generator().manual_seed(9))
    logits2 = pipeline_forward(stage, state, more if rank == 0 else None, 3)
    layer_params = sum(p.numel() for p in stage.layers.parameters())
    return logits, logits2, more, layer_params


def generate_worker(rank, size, cfg_dict, full_sd, prompt, microbatches):
    cfg = ModelConfig(**cfg_dict)
    stage = PipelineStage(cfg, rank, size).eval()
    stage.load_from_full(full_sd)
    return pipeline_generate(stage, prompt, 10, microbatches)


@pytest.mark.parametrize("size,microbatches", [(2, 1), (3, 2), (3, 4)])
def test_pipeline_forward_matches_single_device(size, microbatches):
    cfg = small_cfg()
    ref = make_model(cfg)
    idx = torch.randint(0, 64, (4, 9))
    results = run_distributed(forward_worker, size, cfg.to_dict(), ref.state_dict(), idx, microbatches)

    logits, logits2, more, _ = results[-1]               # the last stage returns the logits
    torch.testing.assert_close(logits, ref(idx), atol=1e-4, rtol=1e-4)
    full = ref(torch.cat([idx, more], dim=1))
    torch.testing.assert_close(logits2, full[:, 9:], atol=1e-4, rtol=1e-4)   # continued through the stage caches
    assert all(r[0] is None for r in results[:-1])
    total = sum(p.numel() for p in ref.layers.parameters())
    assert sum(r[3] for r in results) == total             # every layer lives on exactly one stage


@pytest.mark.parametrize("size,microbatches", [(2, 1), (3, 2)])
def test_pipeline_generation_matches_single_device(size, microbatches):
    cfg = small_cfg()
    ref = make_model(cfg)
    prompt = torch.randint(0, 64, (4, 5))
    results = run_distributed(generate_worker, size, cfg.to_dict(), ref.state_dict(), prompt, microbatches)
    want = generate(ref, prompt, 10)
    for out in results:                                    # every stage ends with the same tokens
        assert torch.equal(out, want)
