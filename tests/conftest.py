import pytest
import torch

from forgeserve import ForgeModel, ModelConfig


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def small_cfg(**kw):
    base = dict(vocab_size=64, d_model=64, n_layers=3, n_heads=8, n_kv_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture
def cfg():
    return small_cfg()


@pytest.fixture
def model(cfg):
    return ForgeModel(cfg).eval()
