import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
RMSNorm = load("01_config_and_primitives", "c_rms_norm").RMSNorm
NemotronHMLP = load("02_layer_types", "a_mlp").NemotronHMLP
from exercise import NemotronHDecoderLayer

HIDDEN_SIZE = 64
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def config():
    return NemotronHConfig(
        vocab_size=256,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_hidden_layers=4,
        hybrid_override_pattern="M-M*",
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
    )


@pytest.fixture
def mlp_mixer(config):
    torch.manual_seed(42)
    return NemotronHMLP(config)


@pytest.fixture
def decoder_layer(config, mlp_mixer):
    return NemotronHDecoderLayer(config, mlp_mixer)


class TestDecoderLayerInit:
    def test_is_nn_module(self, decoder_layer):
        assert isinstance(decoder_layer, torch.nn.Module)

    def test_has_mixer(self, decoder_layer, mlp_mixer):
        assert decoder_layer.mixer is mlp_mixer

    def test_has_norm(self, decoder_layer):
        assert isinstance(decoder_layer.norm, RMSNorm)

    def test_norm_weight_shape(self, decoder_layer):
        assert decoder_layer.norm.weight.shape == (HIDDEN_SIZE,)


class TestDecoderLayerForward:
    def test_returns_tuple(self, decoder_layer):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        result = decoder_layer(x)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self, decoder_layer):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        hidden_states, residual = decoder_layer(x)
        assert hidden_states.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        assert residual.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_first_layer_residual_is_input(self, decoder_layer):
        """When residual=None (first layer), residual should be the original input."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        hidden_states, residual = decoder_layer(x, residual=None)
        assert torch.equal(residual, x)

    def test_subsequent_layer_residual_is_sum(self, decoder_layer):
        """When residual is provided, new residual = old residual + hidden_states."""
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        prev_residual = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        hidden_states, new_residual = decoder_layer(x, residual=prev_residual)
        expected_residual = prev_residual + x
        assert torch.allclose(new_residual, expected_residual, atol=1e-6)


class TestDecoderLayerChaining:
    def test_two_layers_chain(self, config, mlp_mixer):
        """Two decoder layers should chain together naturally."""
        layer1 = NemotronHDecoderLayer(config, NemotronHMLP(config))
        layer2 = NemotronHDecoderLayer(config, NemotronHMLP(config))
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        hidden, residual = layer1(x)
        hidden2, residual2 = layer2(hidden, residual)
        assert hidden2.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        assert residual2.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_gradients_flow_through_chain(self, config):
        layer1 = NemotronHDecoderLayer(config, NemotronHMLP(config))
        layer2 = NemotronHDecoderLayer(config, NemotronHMLP(config))
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, requires_grad=True)
        hidden, residual = layer1(x)
        hidden2, residual2 = layer2(hidden, residual)
        hidden2.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
