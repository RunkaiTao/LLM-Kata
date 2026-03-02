import sys
import math
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
DataLoaderLite = load("02_data_and_tokenization", "b_data_loader_lite").DataLoaderLite
from exercise import train

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
B = 2
T = 16
MAX_STEPS = 50
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def setup():
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )
    model = GPT(config).to(device)
    # Create repeating pattern data so the model can learn
    pattern = torch.arange(VOCAB_SIZE, dtype=torch.long).repeat(20)
    train_loader = DataLoaderLite([pattern], B=B, T=T)
    return model, train_loader


class TestTrainingLoop:
    def test_returns_loss_history(self, setup):
        model, train_loader = setup
        history = train(model, train_loader, MAX_STEPS, LR, device)
        assert isinstance(history, list)
        assert len(history) == MAX_STEPS

    def test_loss_decreases(self, setup):
        model, train_loader = setup
        history = train(model, train_loader, MAX_STEPS, LR, device)
        # Average of first 5 should be higher than average of last 5
        first_avg = sum(history[:5]) / 5
        last_avg = sum(history[-5:]) / 5
        assert last_avg < first_avg, "Training loss should decrease over time"

    def test_all_losses_positive_and_finite(self, setup):
        model, train_loader = setup
        history = train(model, train_loader, MAX_STEPS, LR, device)
        for loss_val in history:
            assert loss_val > 0
            assert math.isfinite(loss_val)

    def test_parameters_change(self, setup):
        model, train_loader = setup
        initial_params = {n: p.clone() for n, p in model.named_parameters()}
        train(model, train_loader, MAX_STEPS, LR, device)
        changed = any(
            not torch.equal(initial_params[n], p)
            for n, p in model.named_parameters()
        )
        assert changed, "Model parameters should change during training"
