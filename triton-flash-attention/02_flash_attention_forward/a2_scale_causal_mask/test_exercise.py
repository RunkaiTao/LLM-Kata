"""Tests for a2_scale_causal_mask."""
import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import scale_causal_mask


class TestScaleCausalMask:
    def test_output_shape(self):
        QK = torch.randn(16, 16, dtype=torch.float32, device="cuda")
        out = scale_causal_mask(QK, softmax_scale=0.5)
        assert out.shape == (16, 16)

    def test_scaling_only(self):
        """Without causal mask, should just multiply by scale."""
        QK = torch.ones(16, 16, dtype=torch.float32, device="cuda") * 4.0
        out = scale_causal_mask(QK, softmax_scale=0.25, causal=False)
        expected = torch.ones(16, 16, device="cuda") * 1.0
        assert torch.allclose(out, expected, atol=1e-5)

    def test_causal_mask_diagonal_block(self):
        """Diagonal block (offset=0): upper triangle should be ~ -1e6."""
        QK = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
        out = scale_causal_mask(QK, softmax_scale=1.0, causal=True,
                                query_offset=0, key_offset=0)
        # Check upper triangle is masked
        for i in range(16):
            for j in range(16):
                if i >= j:
                    assert out[i, j].item() > -1e5, f"({i},{j}) should NOT be masked"
                else:
                    assert out[i, j].item() < -1e5, f"({i},{j}) should be masked"

    def test_causal_mask_left_block(self):
        """Left-of-diagonal block: all queries can see all keys, no masking."""
        QK = torch.ones(16, 16, dtype=torch.float32, device="cuda")
        out = scale_causal_mask(QK, softmax_scale=1.0, causal=True,
                                query_offset=16, key_offset=0)
        # query_pos [16..31] >= key_pos [0..15] always True
        assert (out > -1e5).all()

    def test_causal_mask_right_block(self):
        """Right-of-diagonal block: no queries can see any keys, all masked."""
        QK = torch.ones(16, 16, dtype=torch.float32, device="cuda")
        out = scale_causal_mask(QK, softmax_scale=1.0, causal=True,
                                query_offset=0, key_offset=16)
        # query_pos [0..15] < key_pos [16..31] always
        assert (out < -1e5).all()

    def test_scale_applied_before_mask(self):
        """Visible positions should have value QK * scale, not just QK."""
        QK = torch.full((16, 16), 6.0, dtype=torch.float32, device="cuda")
        out = scale_causal_mask(QK, softmax_scale=0.5, causal=True,
                                query_offset=0, key_offset=0)
        # Diagonal element (0,0): should be 6*0.5 + 0 = 3.0
        assert abs(out[0, 0].item() - 3.0) < 1e-4
        # Last element (15,15): should be 6*0.5 + 0 = 3.0
        assert abs(out[15, 15].item() - 3.0) < 1e-4

    def test_matches_manual_computation(self):
        """Compare against manual torch computation."""
        torch.manual_seed(42)
        QK = torch.randn(16, 32, dtype=torch.float32, device="cuda")
        scale = 0.125
        out = scale_causal_mask(QK, softmax_scale=scale, causal=True,
                                query_offset=4, key_offset=8)
        # Manual
        q_pos = 4 + torch.arange(16, device="cuda")
        k_pos = 8 + torch.arange(32, device="cuda")
        mask = q_pos[:, None] >= k_pos[None, :]
        expected = QK * scale + torch.where(mask, 0.0, -1e6)
        assert torch.allclose(out, expected, atol=1e-4)
