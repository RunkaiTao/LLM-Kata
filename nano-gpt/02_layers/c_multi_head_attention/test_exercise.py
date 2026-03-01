import torch
import pytest
from exercise import MultiHeadAttention

N_EMBD = 32
N_HEAD = 4
HEAD_SIZE = N_EMBD // N_HEAD  # 8
BLOCK_SIZE = 16
BATCH_SIZE = 2
SEQ_LEN = 10


class TestMultiHeadAttention:
    @pytest.fixture
    def mha(self):
        torch.manual_seed(42)
        return MultiHeadAttention(N_EMBD, N_HEAD, HEAD_SIZE, BLOCK_SIZE, dropout=0.0)

    def test_output_shape(self, mha):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = mha(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_parameters_are_registered(self, mha):
        """All head parameters should be reachable via model.parameters()"""
        param_count = sum(p.numel() for p in mha.parameters())
        # Each head: 3 linear layers (n_embd * head_size each, no bias) = 3 * 32 * 8 = 768
        # N_HEAD heads: 4 * 768 = 3072
        # proj: (head_size * n_head) * n_embd + n_embd = 32*32 + 32 = 1056
        expected = N_HEAD * (3 * N_EMBD * HEAD_SIZE) + (HEAD_SIZE * N_HEAD * N_EMBD + N_EMBD)
        assert param_count == expected

    def test_deterministic(self, mha):
        x = torch.randn(1, 5, N_EMBD)
        out1 = mha(x)
        out2 = mha(x)
        assert torch.equal(out1, out2)
