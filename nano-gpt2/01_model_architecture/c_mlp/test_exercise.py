import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import MLP

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 2
SEQ_LEN = 16


@pytest.fixture
def config():
    return GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )


@pytest.fixture
def mlp(config):
    torch.manual_seed(42)
    return MLP(config)


class TestMLP:
    def test_output_shape(self, mlp):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = mlp(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_expansion_factor(self, mlp):
        """c_fc should expand to 4 * n_embd."""
        assert mlp.c_fc.out_features == 4 * N_EMBD
        assert mlp.c_fc.in_features == N_EMBD

    def test_gelu_not_relu(self, mlp):
        """GELU produces small negative values for negative inputs; ReLU would give 0."""
        x = torch.full((1, 1, N_EMBD), -0.5)
        intermediate = mlp.c_fc(x)
        activated = mlp.gelu(intermediate)
        # GELU(-0.5) ≈ -0.154, not zero like ReLU
        neg_mask = intermediate < 0
        if neg_mask.any():
            # At least some negative activations should produce non-zero output
            assert (activated[neg_mask] != 0).any(), "Activation looks like ReLU, not GELU"

    def test_nanogpt_scale_init_flag(self, mlp):
        assert hasattr(mlp.c_proj, "NANOGPT_SCALE_INIT")
        assert mlp.c_proj.NANOGPT_SCALE_INIT == 1

    def test_deterministic(self, mlp):
        x = torch.randn(1, 4, N_EMBD)
        out1 = mlp(x)
        out2 = mlp(x)
        assert torch.equal(out1, out2)
