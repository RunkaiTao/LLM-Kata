import torch
import pytest
from torch.nn import functional as F
from exercise import safe_softmax


class TestSafeSoftmax:
    def test_sum_to_one(self):
        """Output probabilities must sum to 1 along the last dimension."""
        torch.manual_seed(0)
        x = torch.randn(4, 8)
        out = safe_softmax(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-6), f"Rows should sum to 1, got {sums}"

    def test_matches_pytorch(self):
        """Result must be numerically identical to F.softmax."""
        torch.manual_seed(1)
        x = torch.randn(3, 10)
        expected = F.softmax(x, dim=-1)
        out = safe_softmax(x)
        assert torch.allclose(out, expected, atol=1e-6), "Output should match F.softmax"

    def test_no_nan_on_large_values(self):
        """Naive softmax would overflow here; safe softmax must not produce NaN."""
        x = torch.tensor([1000.0, 1001.0, 1002.0])
        out = safe_softmax(x)
        assert not torch.isnan(out).any(), "Output contains NaN on large inputs"
        assert not torch.isinf(out).any(), "Output contains Inf on large inputs"

    def test_output_range(self):
        """All output values must be in (0, 1)."""
        torch.manual_seed(2)
        x = torch.randn(5, 12)
        out = safe_softmax(x)
        assert (out >= 0).all(), "Negative values found"
        assert (out <= 1).all(), "Values greater than 1 found"

    def test_2d_input(self):
        """Should work on a (B, T) shaped batch without errors."""
        torch.manual_seed(3)
        x = torch.randn(2, 6)
        out = safe_softmax(x)
        assert out.shape == (2, 6), f"Expected shape (2, 6), got {out.shape}"
        assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-6)
