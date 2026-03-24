"""Tests for a5_block_ptr_loop."""
import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import block_ptr_loop


class TestBlockPtrLoop:
    def test_output_shape(self):
        X = torch.randn(16, 64, dtype=torch.float32, device="cuda")
        out = block_ptr_loop(X, block_size=16)
        assert out.shape == (16,)

    def test_single_block(self):
        """When COLS == block_size, one iteration should suffice."""
        X = torch.tensor([[1.0, 3.0, 2.0, 0.0],
                          [5.0, 1.0, 4.0, 2.0]], device="cuda")
        out = block_ptr_loop(X, block_size=4)
        expected = torch.tensor([3.0, 5.0], device="cuda")
        assert torch.allclose(out, expected, atol=1e-5)

    def test_two_blocks(self):
        """Row-wise max across two blocks."""
        X = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0, 0.0, 6.0, 1.0],
                          [3.0, 1.0, 0.0, 4.0, 2.0, 7.0, 1.0, 3.0]],
                         device="cuda")
        out = block_ptr_loop(X, block_size=4)
        expected = torch.tensor([6.0, 7.0], device="cuda")
        assert torch.allclose(out, expected, atol=1e-5)

    def test_matches_torch_max(self):
        """Should match torch.max(X, dim=1)."""
        torch.manual_seed(42)
        X = torch.randn(32, 128, dtype=torch.float32, device="cuda")
        out = block_ptr_loop(X, block_size=32)
        expected = X.max(dim=1).values
        assert torch.allclose(out, expected, atol=1e-5)

    def test_different_block_sizes_same_result(self):
        """Result should be independent of block_size."""
        torch.manual_seed(42)
        X = torch.randn(16, 64, dtype=torch.float32, device="cuda")
        out_16 = block_ptr_loop(X, block_size=16)
        out_32 = block_ptr_loop(X, block_size=32)
        out_64 = block_ptr_loop(X, block_size=64)
        assert torch.allclose(out_16, out_32, atol=1e-5)
        assert torch.allclose(out_32, out_64, atol=1e-5)

    def test_negative_values(self):
        """Should handle all-negative values correctly."""
        X = torch.tensor([[-5.0, -3.0, -8.0, -1.0],
                          [-2.0, -9.0, -4.0, -6.0]], device="cuda")
        out = block_ptr_loop(X, block_size=2)
        expected = torch.tensor([-1.0, -2.0], device="cuda")
        assert torch.allclose(out, expected, atol=1e-5)

    def test_many_blocks(self):
        """Should work with many small blocks."""
        torch.manual_seed(7)
        X = torch.randn(16, 256, dtype=torch.float32, device="cuda")
        out = block_ptr_loop(X, block_size=16)
        expected = X.max(dim=1).values
        assert torch.allclose(out, expected, atol=1e-5)
