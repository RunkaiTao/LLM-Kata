import math
import torch
import pytest
from exercise import GPTLanguageModel, estimate_loss

VOCAB_SIZE = 26
BLOCK_SIZE = 8
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 4
EVAL_ITERS = 10
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestEstimateLoss:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        model = GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, dropout=0.0).to(device)
        train_data = torch.randint(0, VOCAB_SIZE, (500,))
        val_data = torch.randint(0, VOCAB_SIZE, (100,))
        return model, train_data, val_data

    def test_returns_dict(self, setup):
        model, train_data, val_data = setup
        result = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        assert isinstance(result, dict)

    def test_has_train_and_val_keys(self, setup):
        model, train_data, val_data = setup
        result = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        assert "train" in result
        assert "val" in result

    def test_losses_are_positive(self, setup):
        model, train_data, val_data = setup
        result = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        assert result["train"] > 0
        assert result["val"] > 0

    def test_model_back_in_train_mode(self, setup):
        """After estimate_loss, model should be back in training mode"""
        model, train_data, val_data = setup
        model.train()
        estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        assert model.training

    def test_no_gradients_accumulated(self, setup):
        """estimate_loss should not accumulate gradients"""
        model, train_data, val_data = setup
        model.zero_grad()
        estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        for param in model.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0

    def test_losses_are_finite(self, setup):
        model, train_data, val_data = setup
        result = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
        assert math.isfinite(result["train"].item())
        assert math.isfinite(result["val"].item())
