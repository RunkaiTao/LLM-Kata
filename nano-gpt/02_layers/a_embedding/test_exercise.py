import torch
import pytest
from exercise import Embeddings

VOCAB_SIZE = 26
BLOCK_SIZE = 16
N_EMBD = 32
BATCH_SIZE = 4
SEQ_LEN = 8


class TestEmbeddings:
    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return Embeddings(VOCAB_SIZE, BLOCK_SIZE, N_EMBD)

    def test_output_shape(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = model(idx)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_different_positions_give_different_embeddings(self, model):
        """Same token at different positions should produce different outputs"""
        idx = torch.zeros((1, 2), dtype=torch.long)  # same token, positions 0 and 1
        out = model(idx)
        assert not torch.equal(out[0, 0], out[0, 1])

    def test_same_input_deterministic(self, model):
        """Same input should give same output"""
        idx = torch.tensor([[5, 3, 1]])
        out1 = model(idx)
        out2 = model(idx)
        assert torch.equal(out1, out2)

    def test_output_dtype(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (2, 4))
        out = model(idx)
        assert out.dtype == torch.float32
