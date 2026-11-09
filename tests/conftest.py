import pytest
import torch

from forgeserve import SINGLE, ForgeModel, ModelConfig


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def small_cfg(**kw):
    base = dict(vocab_size=64, d_model=64, n_layers=3, n_heads=8, n_kv_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def make_model(cfg=None, tp=SINGLE):
    """A model with big enough random weights that greedy generation produces varied tokens.

    With the default tiny init a random model repeats one token forever, which would hide lost or duplicated tokens."""
    model = ForgeModel(cfg or small_cfg(), tp).eval()
    g = torch.Generator().manual_seed(1)
    for p in model.parameters():
        if p.ndim >= 2:
            p.data = torch.randn(p.shape, generator=g) * 0.25
    return model


@pytest.fixture
def cfg():
    return small_cfg()


@pytest.fixture
def model(cfg):
    return make_model(cfg)
