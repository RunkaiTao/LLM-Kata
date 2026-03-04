import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import rotate_half, RotaryEmbedding, apply_rotary_pos_emb

HEAD_DIM = 16
BATCH_SIZE = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
SEQ_LEN = 8


@pytest.fixture
def config():
    return NemotronHConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=4,
        hybrid_override_pattern="M-M*",
        num_attention_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )


@pytest.fixture
def rope(config):
    return RotaryEmbedding(config)


class TestRotateHalf:
    def test_output_shape(self):
        x = torch.randn(2, 4, 8, 16)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotation_values(self):
        """rotate_half([1, 2, 3, 4]) should give [-3, -4, 1, 2]."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        out = rotate_half(x)
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(out, expected)

    def test_double_rotation_negates(self):
        """Applying rotate_half twice should negate the input."""
        x = torch.randn(2, 4)
        out = rotate_half(rotate_half(x))
        assert torch.allclose(out, -x)


class TestRotaryEmbeddingInit:
    def test_is_nn_module(self, rope):
        assert isinstance(rope, torch.nn.Module)

    def test_inv_freq_shape(self, rope):
        assert rope.inv_freq.shape == (HEAD_DIM // 2,)

    def test_inv_freq_decreasing(self, rope):
        """Higher dimension indices should have lower frequencies."""
        assert (rope.inv_freq[:-1] > rope.inv_freq[1:]).all()

    def test_inv_freq_first_is_one(self, rope):
        """First frequency should be 1.0 (base^0 = 1)."""
        assert torch.isclose(rope.inv_freq[0], torch.tensor(1.0))


class TestRotaryEmbeddingForward:
    def test_output_is_tuple(self, rope):
        x = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        cos, sin = rope(x, position_ids)
        assert isinstance(cos, torch.Tensor)
        assert isinstance(sin, torch.Tensor)

    def test_output_shapes(self, rope):
        x = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        cos, sin = rope(x, position_ids)
        assert cos.shape == (BATCH_SIZE, SEQ_LEN, HEAD_DIM)
        assert sin.shape == (BATCH_SIZE, SEQ_LEN, HEAD_DIM)

    def test_position_zero_sin_is_zero(self, rope):
        """At position 0, all angles are 0, so sin should be 0."""
        x = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = rope(x, position_ids)
        assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-6)

    def test_position_zero_cos_is_one(self, rope):
        """At position 0, all angles are 0, so cos should be 1."""
        x = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = rope(x, position_ids)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)

    def test_different_positions_give_different_values(self, rope):
        x = torch.randn(1, NUM_HEADS, 2, HEAD_DIM)
        position_ids = torch.tensor([[0, 5]])
        cos, sin = rope(x, position_ids)
        assert not torch.allclose(cos[:, 0], cos[:, 1])


class TestApplyRotaryPosEmb:
    def test_output_shapes(self, rope):
        q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_position_zero_preserves_input(self, rope):
        """At position 0, rotation should be identity (cos=1, sin=0)."""
        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_rot, q, atol=1e-5)
        assert torch.allclose(k_rot, k, atol=1e-5)

    def test_rotation_changes_values(self, rope):
        """Non-zero positions should change Q and K."""
        q = torch.randn(1, NUM_HEADS, 1, HEAD_DIM)
        k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
        position_ids = torch.tensor([[42]])
        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        assert not torch.allclose(q_rot, q)

    def test_preserves_norm(self, rope):
        """RoPE rotation should preserve the L2 norm of vectors."""
        q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        q_norm = q.norm(dim=-1)
        q_rot_norm = q_rot.norm(dim=-1)
        assert torch.allclose(q_norm, q_rot_norm, atol=1e-4)

    def test_gradients_flow(self, rope):
        q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, requires_grad=True)
        k = torch.randn(BATCH_SIZE, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, requires_grad=True)
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, -1)
        cos, sin = rope(q, position_ids)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        (q_rot.sum() + k_rot.sum()).backward()
        assert q.grad is not None
        assert k.grad is not None
