import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import generate_topk

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )
    return GPT(config).to(device)


class TestTopKGeneration:
    def test_output_length(self, model):
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        result = generate_topk(model, idx, max_new_tokens=10, top_k=50)
        assert result.shape == (1, 11)

    def test_preserves_context(self, model):
        idx = torch.tensor([[5, 3, 1]], device=device)
        result = generate_topk(model, idx, max_new_tokens=5, top_k=50)
        assert torch.equal(result[0, :3], idx[0])

    def test_tokens_in_range(self, model):
        torch.manual_seed(0)
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        result = generate_topk(model, idx, max_new_tokens=30, top_k=50)
        assert (result >= 0).all()
        assert (result < VOCAB_SIZE).all()

    def test_topk_1_is_greedy(self, model):
        """With top_k=1, output should be deterministic (always pick the most likely token)."""
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        torch.manual_seed(0)
        r1 = generate_topk(model, idx.clone(), max_new_tokens=10, top_k=1)
        torch.manual_seed(999)
        r2 = generate_topk(model, idx.clone(), max_new_tokens=10, top_k=1)
        assert torch.equal(r1, r2), "top_k=1 should give deterministic (greedy) output"

    def test_batch_generation(self, model):
        idx = torch.zeros((3, 1), dtype=torch.long, device=device)
        result = generate_topk(model, idx, max_new_tokens=10, top_k=50)
        assert result.shape == (3, 11)

    def test_zero_new_tokens(self, model):
        idx = torch.tensor([[1, 2, 3]], device=device)
        result = generate_topk(model, idx, max_new_tokens=0, top_k=50)
        assert torch.equal(result, idx)
