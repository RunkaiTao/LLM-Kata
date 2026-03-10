import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import NemotronHAttention

HIDDEN_SIZE = 64
NUM_HEADS = 4
HEAD_DIM = 16
NUM_KV_HEADS = 2
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
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_key_value_heads=NUM_KV_HEADS,
    )


@pytest.fixture
def attn(config):
    torch.manual_seed(42)
    return NemotronHAttention(config)


class TestGQAInit:
    def test_is_nn_module(self, attn):
        assert isinstance(attn, torch.nn.Module)

    def test_has_rotary_emb(self, attn):
        assert hasattr(attn, "rotary_emb")

    def test_qkv_proj_output_size(self, attn):
        """QKV projection should output q_size + 2 * kv_size."""
        q_size = NUM_HEADS * HEAD_DIM
        kv_size = NUM_KV_HEADS * HEAD_DIM
        expected = q_size + 2 * kv_size
        assert attn.qkv_proj.out_features == expected

    def test_qkv_proj_input_size(self, attn):
        assert attn.qkv_proj.in_features == HIDDEN_SIZE

    def test_o_proj_shape(self, attn):
        q_size = NUM_HEADS * HEAD_DIM
        assert attn.o_proj.in_features == q_size
        assert attn.o_proj.out_features == HIDDEN_SIZE

    def test_no_bias(self, attn):
        assert attn.qkv_proj.bias is None
        assert attn.o_proj.bias is None

    def test_stored_dimensions(self, attn):
        assert attn.num_heads == NUM_HEADS
        assert attn.num_kv_heads == NUM_KV_HEADS
        assert attn.head_dim == HEAD_DIM
        assert attn.hidden_size == HIDDEN_SIZE

    def test_num_kv_groups(self, attn):
        assert attn.num_kv_groups == NUM_HEADS // NUM_KV_HEADS


class TestGQAForward:
    def test_output_shape(self, attn):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = attn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_causal_masking(self, attn):
        """Changing future tokens should not affect earlier positions."""
        torch.manual_seed(0)
        x1 = torch.randn(1, 4, HIDDEN_SIZE)
        x2 = x1.clone()
        x2[0, 2, :] = torch.randn(HIDDEN_SIZE)
        x2[0, 3, :] = torch.randn(HIDDEN_SIZE)
        out1 = attn(x1)
        out2 = attn(x2)
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)
        assert torch.allclose(out1[0, 1], out2[0, 1], atol=1e-5)

    def test_deterministic(self, attn):
        x = torch.randn(1, 5, HIDDEN_SIZE)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.equal(out1, out2)

    def test_different_sequence_lengths(self, attn):
        for T in [1, 4, 8]:
            x = torch.randn(1, T, HIDDEN_SIZE)
            out = attn(x)
            assert out.shape == (1, T, HIDDEN_SIZE)

    def test_gradients_flow(self, attn):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_position_dependent(self, attn):
        """Output should depend on position (RoPE makes attention position-aware)."""
        x = torch.randn(1, 3, HIDDEN_SIZE)
        out = attn(x)
        # If first and last token had identical input, RoPE should still
        # make their outputs differ (beyond just causal masking differences)
        x_same = torch.zeros(1, 3, HIDDEN_SIZE)
        x_same[0, 0] = x[0, 0]
        x_same[0, 1] = x[0, 0].clone()  # same content, different position
        x_same[0, 2] = x[0, 0].clone()
        out_same = attn(x_same)
        # Position 0 and 1 have same input but different positions -> different output
        assert not torch.allclose(out_same[0, 0], out_same[0, 1], atol=1e-5)


class TestGQAGrouping:
    def test_fewer_kv_than_q_heads(self, attn):
        """GQA has fewer KV heads than Q heads."""
        assert attn.num_kv_heads < attn.num_heads

    def test_q_size_equals_hidden(self, attn):
        """Q total dimension should equal num_heads * head_dim."""
        assert attn.q_size == NUM_HEADS * HEAD_DIM

    def test_kv_size_smaller_than_q(self, attn):
        """KV dimension should be smaller than Q dimension in GQA."""
        assert attn.kv_size < attn.q_size
        assert attn.kv_size == NUM_KV_HEADS * HEAD_DIM
