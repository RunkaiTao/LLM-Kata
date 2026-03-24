"""Tests for a1_matmul_block."""
import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import matmul_block


class TestMatmulBlock:
    def test_output_shape(self):
        """Output shape should be (BLOCK_Q, BLOCK_KV)."""
        Q = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        K_t = torch.randn(64, 16, dtype=torch.float16, device="cuda")
        out = matmul_block(Q, K_t)
        assert out.shape == (16, 16)

    def test_identity_matmul(self):
        """Q @ I should return Q (projected to BLOCK_KV cols)."""
        Q = torch.eye(16, dtype=torch.float16, device="cuda")
        K_t = torch.eye(16, dtype=torch.float16, device="cuda")
        out = matmul_block(Q, K_t)
        expected = torch.eye(16, dtype=torch.float32, device="cuda")
        assert torch.allclose(out, expected, atol=1e-2)

    def test_matches_torch_matmul(self):
        """Result should match torch.matmul within float16 tolerance."""
        torch.manual_seed(42)
        Q = torch.randn(32, 64, dtype=torch.float16, device="cuda")
        K_t = torch.randn(64, 32, dtype=torch.float16, device="cuda")
        out = matmul_block(Q, K_t)
        expected = (Q.float() @ K_t.float())
        assert torch.allclose(out, expected, atol=1e-1), \
            f"Max diff: {(out - expected).abs().max():.4f}"

    def test_asymmetric_shapes(self):
        """BLOCK_Q != BLOCK_KV should work."""
        torch.manual_seed(7)
        Q = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        K_t = torch.randn(64, 32, dtype=torch.float16, device="cuda")
        out = matmul_block(Q, K_t)
        expected = Q.float() @ K_t.float()
        assert out.shape == (16, 32)
        assert torch.allclose(out, expected, atol=1e-1)

    def test_output_dtype_is_float32(self):
        """tl.dot accumulates in float32."""
        Q = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        K_t = torch.randn(64, 16, dtype=torch.float16, device="cuda")
        out = matmul_block(Q, K_t)
        assert out.dtype == torch.float32
