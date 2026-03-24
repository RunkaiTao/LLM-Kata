import sys
from pathlib import Path

import torch
import pytest
from exercise import online_softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

safe_softmax = load("01_attention_fundamentals", "a_safe_softmax").safe_softmax


class TestOnlineSoftmax:
    def test_sum_to_one(self):
        """Output probabilities must sum to 1."""
        torch.manual_seed(0)
        x = torch.randn(8)
        out = online_softmax(x, block_size=4)
        assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6), \
            f"Should sum to 1, got {out.sum()}"

    def test_matches_safe_softmax(self):
        """Must produce the same result as safe_softmax (our reference)."""
        torch.manual_seed(1)
        x = torch.randn(16)
        expected = safe_softmax(x)
        out = online_softmax(x, block_size=4)
        assert torch.allclose(out, expected, atol=1e-6), \
            "online_softmax must match safe_softmax"

    def test_block_size_1(self):
        """block_size=1 is the extreme case: running stats update one element at a time."""
        torch.manual_seed(2)
        x = torch.randn(8)
        expected = safe_softmax(x)
        out = online_softmax(x, block_size=1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_block_size_equals_len(self):
        """Single block: should degenerate to ordinary safe softmax."""
        torch.manual_seed(3)
        x = torch.randn(8)
        expected = safe_softmax(x)
        out = online_softmax(x, block_size=8)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_numerical_stability(self):
        """Must not produce NaN or Inf on large values."""
        x = torch.tensor([1000.0, 1001.0, 1002.0, 999.0])
        out = online_softmax(x, block_size=2)
        assert not torch.isnan(out).any(), "NaN found on large input values"
        assert not torch.isinf(out).any(), "Inf found on large input values"
        assert torch.allclose(out.sum(), torch.tensor(1.0), atol=1e-6)
