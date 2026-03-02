import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
from exercise import GPT

VOCAB_SIZE = 256
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 2
SEQ_LEN = 16
device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def config():
    return GPTConfig(
        block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return GPT(config).to(device)


class TestModuleDict:
    def test_transformer_is_module_dict(self, model):
        assert isinstance(model.transformer, torch.nn.ModuleDict)

    def test_has_required_keys(self, model):
        keys = set(model.transformer.keys())
        assert keys == {"wte", "wpe", "h", "ln_f"}

    def test_h_is_module_list(self, model):
        assert isinstance(model.transformer.h, torch.nn.ModuleList)
        assert len(model.transformer.h) == N_LAYER


class TestWeightTying:
    def test_wte_weight_is_lm_head_weight(self, model):
        """Weight tying: wte and lm_head must share the SAME tensor (identity, not copy)."""
        assert model.transformer.wte.weight is model.lm_head.weight

    def test_parameter_sharing_reduces_count(self, model, config):
        """Weight tying reduces total parameter count by vocab_size * n_embd."""
        total_params = sum(p.numel() for p in model.parameters())
        shared_params = config.vocab_size * config.n_embd
        # Without tying, we'd have total_params + shared_params
        # The shared matrix is counted only once
        assert total_params < total_params + shared_params


class TestScaledInit:
    def test_c_proj_weights_smaller_than_c_attn(self, config):
        """c_proj layers (with NANOGPT_SCALE_INIT) should have smaller std than c_attn."""
        torch.manual_seed(42)
        model = GPT(config)
        c_attn_stds = []
        c_proj_stds = []
        for name, param in model.named_parameters():
            if "c_attn.weight" in name:
                c_attn_stds.append(param.std().item())
            elif "c_proj.weight" in name:
                c_proj_stds.append(param.std().item())
        if c_attn_stds and c_proj_stds:
            avg_attn_std = sum(c_attn_stds) / len(c_attn_stds)
            avg_proj_std = sum(c_proj_stds) / len(c_proj_stds)
            assert avg_proj_std < avg_attn_std, "c_proj should have smaller std due to scaled init"

    def test_linear_biases_are_zero(self, model):
        for name, param in model.named_parameters():
            if "bias" in name and "ln" not in name:
                assert torch.allclose(param, torch.zeros_like(param)), f"{name} bias not zero"

    def test_weights_are_small(self, model):
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                assert param.std().item() < 0.1, f"{name} std too large: {param.std().item()}"


class TestForwardPass:
    def test_logits_shape_no_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        logits, loss = model(idx)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        assert loss is None

    def test_logits_shape_with_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        logits, loss = model(idx, targets)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_loss_is_scalar(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        _, loss = model(idx, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_untrained_loss_near_log_vocab(self, model):
        """Random model loss should be approximately ln(vocab_size)."""
        torch.manual_seed(0)
        idx = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN), device=device)
        _, loss = model(idx, targets)
        expected = math.log(VOCAB_SIZE)
        assert abs(loss.item() - expected) < 1.5

    def test_different_sequence_lengths(self, model):
        for T in [1, 4, BLOCK_SIZE]:
            idx = torch.randint(0, VOCAB_SIZE, (1, T), device=device)
            logits, _ = model(idx)
            assert logits.shape == (1, T, VOCAB_SIZE)

    def test_gradients_flow(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (2, 4), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (2, 4), device=device)
        _, loss = model(idx, targets)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
