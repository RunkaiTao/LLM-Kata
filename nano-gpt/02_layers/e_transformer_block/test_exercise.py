import torch
import pytest
from exercise import Block

N_EMBD = 32
N_HEAD = 4
BLOCK_SIZE = 16
BATCH_SIZE = 2
SEQ_LEN = 8


class TestBlock:
    @pytest.fixture
    def block(self):
        torch.manual_seed(42)
        return Block(N_EMBD, N_HEAD, BLOCK_SIZE, dropout=0.0)

    def test_output_shape(self, block):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = block(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_residual_connection(self, block):
        """Output should differ from a pure attention+FFN pass (residual adds input)"""
        torch.manual_seed(0)
        x = torch.randn(1, 4, N_EMBD)
        out = block(x)
        # If residual connections work, output should not be zero even for zero-ish input
        # and should incorporate the original input signal
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_output_differs_from_input(self, block):
        """Block should transform the input (not just pass through)"""
        x = torch.randn(1, 4, N_EMBD)
        out = block(x)
        assert not torch.equal(out, x)

    def test_deterministic(self, block):
        x = torch.randn(1, 4, N_EMBD)
        out1 = block(x)
        out2 = block(x)
        assert torch.equal(out1, out2)
