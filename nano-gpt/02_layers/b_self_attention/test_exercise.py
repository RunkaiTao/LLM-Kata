import torch
import pytest
from exercise import Head

N_EMBD = 32
HEAD_SIZE = 8
BLOCK_SIZE = 16
BATCH_SIZE = 2
SEQ_LEN = 10


class TestHead:
    @pytest.fixture
    def head(self):
        torch.manual_seed(42)
        return Head(N_EMBD, HEAD_SIZE, BLOCK_SIZE, dropout=0.0)

    def test_output_shape(self, head):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = head(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HEAD_SIZE)

    def test_causal_masking_works(self, head):
        """
        Position 0 should not attend to later positions.
        Changing tokens at positions 1 and 2 should not affect output at position 0.
        """
        torch.manual_seed(0)
        x1 = torch.randn(1, 3, N_EMBD)
        x2 = x1.clone()
        x2[0, 1, :] = torch.randn(N_EMBD)
        x2[0, 2, :] = torch.randn(N_EMBD)
        out1 = head(x1)
        out2 = head(x2)
        # Position 0 output should be identical
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)

    def test_attention_weights_sum_to_one(self, head):
        """After softmax, attention weights along each row should sum to ~1"""
        x = torch.randn(1, 4, N_EMBD)
        k = head.key(x)
        q = head.query(x)
        wei = q @ k.transpose(-2, -1) * (HEAD_SIZE**-0.5)
        wei = wei.masked_fill(head.tril[:4, :4] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        row_sums = wei.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_deterministic_output(self, head):
        """Same input should produce same output"""
        x = torch.randn(1, 5, N_EMBD)
        out1 = head(x)
        out2 = head(x)
        assert torch.equal(out1, out2)
