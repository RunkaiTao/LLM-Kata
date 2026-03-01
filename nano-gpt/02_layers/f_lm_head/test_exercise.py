import torch
import pytest
from exercise import LMHead

N_EMBD = 32
VOCAB_SIZE = 26
BATCH_SIZE = 2
SEQ_LEN = 8


class TestLMHead:
    @pytest.fixture
    def head(self):
        torch.manual_seed(42)
        return LMHead(N_EMBD, VOCAB_SIZE)

    def test_output_shape(self, head):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = head(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_output_differs_from_input(self, head):
        """LMHead should transform the input (different dimensions)"""
        x = torch.randn(1, 4, N_EMBD)
        out = head(x)
        assert out.shape[-1] == VOCAB_SIZE
        assert out.shape[-1] != x.shape[-1]

    def test_deterministic(self, head):
        x = torch.randn(1, 4, N_EMBD)
        out1 = head(x)
        out2 = head(x)
        assert torch.equal(out1, out2)

    def test_output_dtype(self, head):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = head(x)
        assert out.dtype == torch.float32
