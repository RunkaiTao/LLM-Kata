import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import configure_optimizers

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2


@pytest.fixture
def model():
    torch.manual_seed(42)
    config = GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )
    return GPT(config)


class TestConfigureOptimizers:
    def test_returns_adamw(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        assert isinstance(opt, torch.optim.AdamW)

    def test_two_param_groups(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        assert len(opt.param_groups) == 2

    def test_decay_group_has_weight_decay(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        decay_group = opt.param_groups[0]
        assert decay_group["weight_decay"] == 0.1

    def test_no_decay_group_has_zero_weight_decay(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        nodecay_group = opt.param_groups[1]
        assert nodecay_group["weight_decay"] == 0.0

    def test_all_params_accounted_for(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        opt_params = sum(len(g["params"]) for g in opt.param_groups)
        model_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert opt_params == model_params

    def test_2d_params_in_decay_group(self, model):
        """All parameters in the decay group should be >= 2D (weight matrices)."""
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        decay_group = opt.param_groups[0]
        for p in decay_group["params"]:
            assert p.dim() >= 2, f"Found {p.dim()}D param in decay group"

    def test_1d_params_in_no_decay_group(self, model):
        """All parameters in the no-decay group should be < 2D (biases, norms)."""
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        nodecay_group = opt.param_groups[1]
        for p in nodecay_group["params"]:
            assert p.dim() < 2, f"Found {p.dim()}D param in no-decay group"

    def test_betas(self, model):
        opt = configure_optimizers(model, weight_decay=0.1, learning_rate=6e-4, device_type="cpu")
        for group in opt.param_groups:
            assert group["betas"] == (0.9, 0.95)
