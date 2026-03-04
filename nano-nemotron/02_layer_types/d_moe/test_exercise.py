import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
NemotronHMLP = load("02_layer_types", "a_mlp").NemotronHMLP
from exercise import NemotronHMoE

HIDDEN_SIZE = 64
N_ROUTED_EXPERTS = 4
N_SHARED_EXPERTS = 1
MOE_INTERMEDIATE_SIZE = 128
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE = 256
NUM_EXPERTS_PER_TOK = 2
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
        n_routed_experts=N_ROUTED_EXPERTS,
        n_shared_experts=N_SHARED_EXPERTS,
        moe_intermediate_size=MOE_INTERMEDIATE_SIZE,
        moe_shared_expert_intermediate_size=MOE_SHARED_EXPERT_INTERMEDIATE_SIZE,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        routed_scaling_factor=1.0,
    )


@pytest.fixture
def moe(config):
    torch.manual_seed(42)
    return NemotronHMoE(config)


class TestMoEInit:
    def test_is_nn_module(self, moe):
        assert isinstance(moe, torch.nn.Module)

    def test_gate_shape(self, moe):
        """Router should project hidden_size -> n_routed_experts."""
        assert moe.gate.in_features == HIDDEN_SIZE
        assert moe.gate.out_features == N_ROUTED_EXPERTS

    def test_gate_no_bias(self, moe):
        assert moe.gate.bias is None

    def test_shared_experts_exists(self, moe):
        assert moe.shared_experts is not None
        assert isinstance(moe.shared_experts, NemotronHMLP)

    def test_shared_expert_intermediate_size(self, moe):
        """Shared expert uses moe_shared_expert_intermediate_size from config."""
        assert moe.shared_experts.up_proj.out_features == MOE_SHARED_EXPERT_INTERMEDIATE_SIZE

    def test_experts_count(self, moe):
        assert isinstance(moe.experts, torch.nn.ModuleList)
        assert len(moe.experts) == N_ROUTED_EXPERTS

    def test_expert_intermediate_size(self, moe):
        for expert in moe.experts:
            assert expert.up_proj.out_features == MOE_INTERMEDIATE_SIZE

    def test_stored_config(self, moe):
        assert moe.n_routed_experts == N_ROUTED_EXPERTS
        assert moe.num_experts_per_tok == NUM_EXPERTS_PER_TOK
        assert moe.routed_scaling_factor == 1.0


class TestMoEForward:
    def test_output_shape_3d(self, moe):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = moe(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_output_shape_2d(self, moe):
        x = torch.randn(4, HIDDEN_SIZE)
        out = moe(x)
        assert out.shape == (4, HIDDEN_SIZE)

    def test_deterministic(self, moe):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out1 = moe(x)
        out2 = moe(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_gradients_flow(self, moe):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestMoERouting:
    def test_sparse_computation(self, moe):
        """Only num_experts_per_tok experts should contribute per token."""
        # The output should be non-zero (since shared expert always fires)
        x = torch.randn(1, 1, HIDDEN_SIZE)
        out = moe(x)
        assert out.abs().sum() > 0

    def test_different_tokens_can_route_differently(self, moe):
        """Different tokens may route to different experts."""
        torch.manual_seed(123)
        x = torch.randn(1, 4, HIDDEN_SIZE)
        # We just verify it doesn't crash with multiple tokens
        out = moe(x)
        assert out.shape == (1, 4, HIDDEN_SIZE)
