import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
ManualAttention = load("01_model_architecture", "b_causal_self_attention").CausalSelfAttention
from exercise import CausalSelfAttention as FlashAttention

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
def flash_attn(config):
    torch.manual_seed(42)
    return FlashAttention(config)


class TestFlashAttention:
    def test_output_shape(self, flash_attn):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = flash_attn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_no_bias_buffer(self, flash_attn):
        """Flash attention should NOT register a 'bias' buffer."""
        buffer_names = [name for name, _ in flash_attn.named_buffers()]
        assert "bias" not in buffer_names, "Flash attention should not have a causal mask buffer"

    def test_matches_manual_attention(self, config):
        """Flash attention output should match manual attention (same weights)."""
        torch.manual_seed(42)
        manual = ManualAttention(config)
        torch.manual_seed(42)
        flash = FlashAttention(config)
        # Copy weights from manual to flash (they should be same from seed, but be safe)
        flash.load_state_dict(manual.state_dict(), strict=False)
        x = torch.randn(1, 8, N_EMBD)
        out_manual = manual(x)
        out_flash = flash(x)
        assert torch.allclose(out_manual, out_flash, atol=1e-5), \
            f"Max diff: {(out_manual - out_flash).abs().max().item()}"

    def test_causal_masking_still_works(self, flash_attn):
        """Changing future tokens should not affect earlier positions."""
        torch.manual_seed(0)
        x1 = torch.randn(1, 4, N_EMBD)
        x2 = x1.clone()
        x2[0, 2, :] = torch.randn(N_EMBD)
        x2[0, 3, :] = torch.randn(N_EMBD)
        out1 = flash_attn(x1)
        out2 = flash_attn(x2)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)
        assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-6)

    def test_gradients_flow(self, flash_attn):
        x = torch.randn(1, 4, N_EMBD, requires_grad=True)
        out = flash_attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
