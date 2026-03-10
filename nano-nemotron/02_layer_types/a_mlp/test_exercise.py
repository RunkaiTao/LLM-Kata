import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import NemotronHMLP

HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 256
BATCH_SIZE = 2
SEQ_LEN = 16


@pytest.fixture
def config():
    return NemotronHConfig(
        vocab_size=256,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=4,
        hybrid_override_pattern="M-M*",
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
    )


@pytest.fixture
def mlp(config):
    torch.manual_seed(42)
    return NemotronHMLP(config)


class TestMLPInit:
    def test_is_nn_module(self, mlp):
        assert isinstance(mlp, torch.nn.Module)

    def test_up_proj_shape(self, mlp):
        assert mlp.up_proj.in_features == HIDDEN_SIZE
        assert mlp.up_proj.out_features == INTERMEDIATE_SIZE

    def test_down_proj_shape(self, mlp):
        assert mlp.down_proj.in_features == INTERMEDIATE_SIZE
        assert mlp.down_proj.out_features == HIDDEN_SIZE

    def test_no_bias_up_proj(self, mlp):
        assert mlp.up_proj.bias is None

    def test_no_bias_down_proj(self, mlp):
        assert mlp.down_proj.bias is None

    def test_has_relu_squared_activation(self, mlp):
        ReLUSquaredActivation = load("01_config_and_primitives", "b_relu_squared").ReLUSquaredActivation
        assert isinstance(mlp.act_fn, ReLUSquaredActivation)


class TestMLPForward:
    def test_output_shape_3d(self, mlp):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mlp(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_output_shape_2d(self, mlp):
        x = torch.randn(4, HIDDEN_SIZE)
        out = mlp(x)
        assert out.shape == (4, HIDDEN_SIZE)

    def test_deterministic(self, mlp):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out1 = mlp(x)
        out2 = mlp(x)
        assert torch.equal(out1, out2)

    def test_gradients_flow(self, mlp):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMLPCustomSizes:
    def test_custom_hidden_intermediate(self, config):
        mlp = NemotronHMLP(config, hidden_size=32, intermediate_size=128)
        assert mlp.up_proj.in_features == 32
        assert mlp.up_proj.out_features == 128
        assert mlp.down_proj.in_features == 128
        assert mlp.down_proj.out_features == 32
        x = torch.randn(2, 8, 32)
        out = mlp(x)
        assert out.shape == (2, 8, 32)
