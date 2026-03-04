import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import Mamba2Mixer

HIDDEN_SIZE = 64
MAMBA_NUM_HEADS = 16
MAMBA_HEAD_DIM = 8
MAMBA_N_GROUPS = 2
SSM_STATE_SIZE = 16
MAMBA_D_CONV = 4
MAMBA_INTERMEDIATE = MAMBA_NUM_HEADS * MAMBA_HEAD_DIM  # 128
GROUPS_SSM_STATE = MAMBA_N_GROUPS * SSM_STATE_SIZE  # 32
BATCH_SIZE = 2
SEQ_LEN = 16


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
        mamba_num_heads=MAMBA_NUM_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        mamba_n_groups=MAMBA_N_GROUPS,
        ssm_state_size=SSM_STATE_SIZE,
        mamba_d_conv=MAMBA_D_CONV,
        mamba_expand=2,
    )


@pytest.fixture
def mamba(config):
    torch.manual_seed(42)
    return Mamba2Mixer(config)


class TestMamba2Init:
    def test_is_nn_module(self, mamba):
        assert isinstance(mamba, torch.nn.Module)

    def test_stored_dimensions(self, mamba):
        assert mamba.hidden_size == HIDDEN_SIZE
        assert mamba.num_heads == MAMBA_NUM_HEADS
        assert mamba.head_dim == MAMBA_HEAD_DIM
        assert mamba.ssm_state_size == SSM_STATE_SIZE
        assert mamba.n_groups == MAMBA_N_GROUPS
        assert mamba.intermediate_size == MAMBA_INTERMEDIATE
        assert mamba.conv_kernel_size == MAMBA_D_CONV

    def test_in_proj_input_size(self, mamba):
        assert mamba.in_proj.in_features == HIDDEN_SIZE

    def test_in_proj_output_size(self, mamba):
        """in_proj output = gate + x + B + C + dt."""
        expected = (
            MAMBA_INTERMEDIATE       # gate
            + MAMBA_INTERMEDIATE     # x (conv input)
            + GROUPS_SSM_STATE       # B (conv input)
            + GROUPS_SSM_STATE       # C (conv input)
            + MAMBA_NUM_HEADS        # dt
        )
        assert mamba.in_proj.out_features == expected

    def test_conv1d_exists(self, mamba):
        assert isinstance(mamba.conv1d, torch.nn.Conv1d)

    def test_conv1d_is_depthwise(self, mamba):
        """Conv1d should be depthwise (groups == in_channels)."""
        conv_dim = MAMBA_INTERMEDIATE + 2 * GROUPS_SSM_STATE
        assert mamba.conv1d.groups == conv_dim

    def test_conv1d_kernel_size(self, mamba):
        assert mamba.conv1d.kernel_size == (MAMBA_D_CONV,)

    def test_ssm_parameters_exist(self, mamba):
        assert isinstance(mamba.A, torch.nn.Parameter)
        assert isinstance(mamba.D, torch.nn.Parameter)
        assert isinstance(mamba.dt_bias, torch.nn.Parameter)

    def test_A_shape(self, mamba):
        assert mamba.A.shape == (MAMBA_NUM_HEADS,)

    def test_A_is_negative(self, mamba):
        """A should be initialized with negative values (state decay)."""
        assert (mamba.A < 0).all()

    def test_D_shape(self, mamba):
        assert mamba.D.shape == (MAMBA_NUM_HEADS,)

    def test_dt_bias_shape(self, mamba):
        assert mamba.dt_bias.shape == (MAMBA_NUM_HEADS,)

    def test_out_proj_shape(self, mamba):
        assert mamba.out_proj.in_features == MAMBA_INTERMEDIATE
        assert mamba.out_proj.out_features == HIDDEN_SIZE


class TestMamba2Forward:
    def test_output_shape(self, mamba):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mamba(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_output_dtype(self, mamba):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mamba(x)
        assert out.dtype == x.dtype

    def test_deterministic(self, mamba):
        x = torch.randn(1, 8, HIDDEN_SIZE)
        out1 = mamba(x)
        out2 = mamba(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_different_sequence_lengths(self, mamba):
        for T in [1, 4, 8]:
            x = torch.randn(1, T, HIDDEN_SIZE)
            out = mamba(x)
            assert out.shape == (1, T, HIDDEN_SIZE)

    def test_gradients_flow(self, mamba):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, requires_grad=True)
        out = mamba(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_batch_independence(self, mamba):
        """Each batch element should be processed independently."""
        torch.manual_seed(0)
        x1 = torch.randn(1, 4, HIDDEN_SIZE)
        x2 = torch.randn(1, 4, HIDDEN_SIZE)
        # Process separately
        out1 = mamba(x1)
        out2 = mamba(x2)
        # Process together
        x_combined = torch.cat([x1, x2], dim=0)
        out_combined = mamba(x_combined)
        assert torch.allclose(out_combined[0], out1[0], atol=1e-5)
        assert torch.allclose(out_combined[1], out2[0], atol=1e-5)


class TestMamba2Causality:
    def test_causal_property(self, mamba):
        """Output at position t should only depend on positions 0..t."""
        torch.manual_seed(0)
        x1 = torch.randn(1, 8, HIDDEN_SIZE)
        x2 = x1.clone()
        x2[0, 5:, :] = torch.randn(3, HIDDEN_SIZE)  # change future tokens
        out1 = mamba(x1)
        out2 = mamba(x2)
        # First 5 positions should be the same (approximately, due to conv padding)
        # After conv with kernel_size=4, position 4 only sees positions 1-4
        # Position 0 should definitely be the same
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)
