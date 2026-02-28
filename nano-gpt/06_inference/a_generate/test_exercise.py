import torch
import pytest
from exercise import GPTLanguageModel

VOCAB_SIZE = 26
BLOCK_SIZE = 16
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestGenerate:
    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, dropout=0.0).to(device)

    def test_output_length(self, model):
        """Generated sequence should have T + max_new_tokens columns"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        max_new = 10
        result = model.generate(idx, max_new)
        assert result.shape == (1, 1 + max_new)

    def test_output_preserves_context(self, model):
        """The original context tokens should be preserved at the start"""
        idx = torch.tensor([[5, 3, 1]], device=device)
        result = model.generate(idx, 5)
        assert torch.equal(result[0, :3], idx[0])

    def test_generated_tokens_in_range(self, model):
        """All generated tokens should be valid token IDs in [0, vocab_size)"""
        torch.manual_seed(0)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        result = model.generate(idx, 50)
        assert (result >= 0).all()
        assert (result < VOCAB_SIZE).all()

    def test_batch_generation(self, model):
        """Generation should work with batch_size > 1"""
        idx = torch.zeros((3, 1), dtype=torch.long, device=device)
        result = model.generate(idx, 10)
        assert result.shape == (3, 11)

    def test_handles_long_context(self, model):
        """
        When context exceeds block_size, it should still work
        (cropping to last block_size tokens internally).
        """
        idx = torch.zeros((1, BLOCK_SIZE + 5), dtype=torch.long, device=device)
        result = model.generate(idx, 5)
        assert result.shape == (1, BLOCK_SIZE + 5 + 5)

    def test_stochastic_generation(self, model):
        """Different seeds should produce different outputs (usually)"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        torch.manual_seed(0)
        r1 = model.generate(idx.clone(), 20)
        torch.manual_seed(999)
        r2 = model.generate(idx.clone(), 20)
        assert not torch.equal(r1, r2)

    def test_deterministic_with_same_seed(self, model):
        """Same seed should produce identical output"""
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        torch.manual_seed(123)
        r1 = model.generate(idx.clone(), 20)
        torch.manual_seed(123)
        r2 = model.generate(idx.clone(), 20)
        assert torch.equal(r1, r2)

    def test_zero_new_tokens(self, model):
        """Generating 0 new tokens should return the original context"""
        idx = torch.tensor([[1, 2, 3]], device=device)
        result = model.generate(idx, 0)
        assert torch.equal(result, idx)
