import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import Block

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
def block(config):
    torch.manual_seed(42)
    return Block(config)


class TestBlock:
    def test_output_shape(self, block):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = block(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_residual_connection(self, block):
        """Output should include the original input signal (residual)."""
        torch.manual_seed(0)
        x = torch.randn(1, 4, N_EMBD)
        out = block(x)
        # If residual works, output should correlate with input
        # (not just be a completely independent transformation)
        diff = (out - x).abs().mean()
        # The difference should be finite and the output should not equal input
        assert diff > 0, "Block did nothing (no transformation)"
        assert diff < 100, "Residual connection seems broken (output too far from input)"

    def test_output_differs_from_input(self, block):
        """The block should actually transform the input (not be identity)."""
        x = torch.randn(1, 4, N_EMBD)
        out = block(x)
        assert not torch.equal(out, x)

    def test_deterministic(self, block):
        x = torch.randn(1, 5, N_EMBD)
        out1 = block(x)
        out2 = block(x)
        assert torch.equal(out1, out2)

    def test_has_pre_norm_components(self, block):
        """Block should have ln_1, attn, ln_2, mlp attributes."""
        assert hasattr(block, "ln_1")
        assert hasattr(block, "attn")
        assert hasattr(block, "ln_2")
        assert hasattr(block, "mlp")
