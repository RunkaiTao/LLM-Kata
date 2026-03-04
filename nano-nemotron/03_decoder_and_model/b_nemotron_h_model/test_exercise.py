import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import NemotronHModel

HIDDEN_SIZE = 64
VOCAB_SIZE = 256
NUM_LAYERS = 6
HYBRID_PATTERN = "M-M*-E"
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def config():
    return NemotronHConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        hybrid_override_pattern=HYBRID_PATTERN,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        mamba_num_heads=16,
        mamba_head_dim=8,
        mamba_n_groups=2,
        ssm_state_size=16,
        mamba_d_conv=4,
        mamba_expand=2,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return NemotronHModel(config)


class TestModelInit:
    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_has_embed_tokens(self, model):
        assert isinstance(model.embed_tokens, torch.nn.Embedding)
        assert model.embed_tokens.num_embeddings == VOCAB_SIZE
        assert model.embed_tokens.embedding_dim == HIDDEN_SIZE

    def test_has_correct_num_layers(self, model):
        assert isinstance(model.layers, torch.nn.ModuleList)
        assert len(model.layers) == NUM_LAYERS

    def test_has_final_norm(self, model):
        RMSNorm = load("01_config_and_primitives", "c_rms_norm").RMSNorm
        assert isinstance(model.norm_f, RMSNorm)


class TestHybridPattern:
    def test_layer_types_match_pattern(self, model):
        """Each layer's mixer type should match the pattern character."""
        NemotronHMLP = load("02_layer_types", "a_mlp").NemotronHMLP
        NemotronHAttention = load("02_layer_types", "b_gqa_attention").NemotronHAttention
        Mamba2Mixer = load("02_layer_types", "c_mamba2").Mamba2Mixer
        NemotronHMoE = load("02_layer_types", "d_moe").NemotronHMoE

        expected_types = {
            "M": Mamba2Mixer,
            "-": NemotronHMLP,
            "*": NemotronHAttention,
            "E": NemotronHMoE,
        }

        for i, char in enumerate(HYBRID_PATTERN):
            mixer = model.layers[i].mixer
            expected = expected_types[char]
            assert isinstance(mixer, expected), (
                f"Layer {i} (pattern '{char}'): expected {expected.__name__}, "
                f"got {type(mixer).__name__}"
            )


class TestModelForward:
    def test_output_shape(self, model):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = model(input_ids)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_output_dtype(self, model):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = model(input_ids)
        assert out.dtype == torch.float32

    def test_deterministic(self, model):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_different_seq_lengths(self, model):
        for T in [1, 4, 8]:
            input_ids = torch.randint(0, VOCAB_SIZE, (1, T))
            out = model(input_ids)
            assert out.shape == (1, T, HIDDEN_SIZE)

    def test_gradients_flow(self, model):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        out = model(input_ids)
        out.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
