import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import CausalSelfAttention

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
def attn(config):
    torch.manual_seed(42)
    return CausalSelfAttention(config)


class TestCausalSelfAttention:
    def test_output_shape(self, attn):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = attn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_causal_masking(self, attn):
        """Changing future tokens should not affect earlier positions."""
        torch.manual_seed(0)
        x1 = torch.randn(1, 4, N_EMBD)
        x2 = x1.clone()
        x2[0, 2, :] = torch.randn(N_EMBD)
        x2[0, 3, :] = torch.randn(N_EMBD)
        out1 = attn(x1)
        out2 = attn(x2)
        # Positions 0 and 1 should be identical
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-6)

    def test_c_attn_is_batched(self, attn):
        """c_attn should project to 3 * n_embd (Q, K, V concatenated)."""
        assert attn.c_attn.out_features == 3 * N_EMBD
        assert attn.c_attn.in_features == N_EMBD

    def test_nanogpt_scale_init_flag(self, attn):
        """c_proj should have NANOGPT_SCALE_INIT attribute set to 1."""
        assert hasattr(attn.c_proj, "NANOGPT_SCALE_INIT")
        assert attn.c_proj.NANOGPT_SCALE_INIT == 1

    def test_deterministic_output(self, attn):
        x = torch.randn(1, 5, N_EMBD)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.equal(out1, out2)

    def test_different_sequence_lengths(self, attn):
        """Should work for any T <= block_size."""
        for T in [1, 4, BLOCK_SIZE]:
            x = torch.randn(1, T, N_EMBD)
            out = attn(x)
            assert out.shape == (1, T, N_EMBD)
