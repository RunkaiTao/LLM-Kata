import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
DataLoaderLite = load("02_data_and_tokenization", "b_data_loader_lite").DataLoaderLite
from exercise import estimate_val_loss

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
B = 2
T = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def setup():
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )
    model = GPT(config).to(device)
    val_data = torch.arange(VOCAB_SIZE, dtype=torch.long).repeat(20)
    val_loader = DataLoaderLite([val_data], B=B, T=T)
    return model, val_loader


class TestValidationLoss:
    def test_returns_positive_float(self, setup):
        model, val_loader = setup
        loss = estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_is_finite(self, setup):
        model, val_loader = setup
        loss = estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        assert math.isfinite(loss)

    def test_no_gradients_accumulated(self, setup):
        model, val_loader = setup
        # Clear any existing gradients
        model.zero_grad()
        estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        # No parameters should have gradients after eval
        for p in model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_model_back_to_train_mode(self, setup):
        model, val_loader = setup
        model.train()  # ensure training mode
        estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        assert model.training, "Model should be back in training mode after eval"

    def test_loss_near_log_vocab_for_random_model(self, setup):
        """Untrained model loss should be approximately ln(vocab_size)."""
        model, val_loader = setup
        loss = estimate_val_loss(model, val_loader, val_loss_steps=10, device=device)
        expected = math.log(VOCAB_SIZE)
        assert abs(loss - expected) < 1.5

    def test_loader_resets(self, setup):
        """Calling estimate_val_loss twice should give same result (loader resets)."""
        model, val_loader = setup
        loss1 = estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        loss2 = estimate_val_loss(model, val_loader, val_loss_steps=5, device=device)
        assert abs(loss1 - loss2) < 1e-5
