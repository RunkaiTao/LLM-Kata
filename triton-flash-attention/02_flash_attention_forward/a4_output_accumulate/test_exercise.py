"""Tests for a4_output_accumulate."""
import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import output_accumulate


class TestOutputAccumulate:
    def test_output_shape(self):
        O = torch.zeros(16, 64, dtype=torch.float32, device="cuda")
        P = torch.randn(16, 16, dtype=torch.float32, device="cuda")
        V = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        alpha = torch.ones(16, dtype=torch.float32, device="cuda")
        out = output_accumulate(O, P, V, alpha)
        assert out.shape == (16, 64)
        assert out.dtype == torch.float32

    def test_first_block_alpha_zero(self):
        """alpha=0 should zero out old O, result = P @ V only."""
        O = torch.ones(16, 64, dtype=torch.float32, device="cuda") * 999
        P = torch.eye(16, dtype=torch.float32, device="cuda")
        V = torch.randn(16, 64, dtype=torch.float16, device="cuda")
        alpha = torch.zeros(16, dtype=torch.float32, device="cuda")
        out = output_accumulate(O, P, V, alpha)
        expected = (P.half() @ V).float()
        assert torch.allclose(out, expected, atol=1e-1)

    def test_alpha_one_adds(self):
        """alpha=1 means O_new = O_old + P @ V."""
        O = torch.ones(16, 64, dtype=torch.float32, device="cuda")
        P = torch.eye(16, dtype=torch.float32, device="cuda")
        V = torch.ones(16, 64, dtype=torch.float16, device="cuda") * 2
        alpha = torch.ones(16, dtype=torch.float32, device="cuda")
        out = output_accumulate(O, P, V, alpha)
        # O_old * 1 + I @ (2*ones) = 1 + 2 = 3
        expected = torch.full((16, 64), 3.0, device="cuda")
        assert torch.allclose(out, expected, atol=1e-1)

    def test_rescaling(self):
        """alpha=0.5 should halve old output before adding."""
        O = torch.full((16, 64), 10.0, dtype=torch.float32, device="cuda")
        P = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
        V = torch.zeros(16, 64, dtype=torch.float16, device="cuda")
        alpha = torch.full((16,), 0.5, dtype=torch.float32, device="cuda")
        out = output_accumulate(O, P, V, alpha)
        # O * 0.5 + 0 = 5
        expected = torch.full((16, 64), 5.0, device="cuda")
        assert torch.allclose(out, expected, atol=1e-4)

    def test_per_row_alpha(self):
        """Each row should be rescaled by its own alpha."""
        O = torch.full((16, 64), 10.0, dtype=torch.float32, device="cuda")
        P = torch.zeros(16, 16, dtype=torch.float32, device="cuda")
        V = torch.zeros(16, 64, dtype=torch.float16, device="cuda")
        alpha = torch.zeros(16, dtype=torch.float32, device="cuda")
        alpha[0] = 1.0   # row 0 keeps its value
        alpha[1] = 0.5   # row 1 gets halved
        out = output_accumulate(O, P, V, alpha)
        assert torch.allclose(out[0], torch.full((64,), 10.0, device="cuda"), atol=1e-4)
        assert torch.allclose(out[1], torch.full((64,), 5.0, device="cuda"), atol=1e-4)
        assert torch.allclose(out[2], torch.zeros(64, device="cuda"), atol=1e-4)

    def test_matches_manual(self):
        """Compare against manual torch computation."""
        torch.manual_seed(42)
        O = torch.randn(16, 64, dtype=torch.float32, device="cuda")
        P = torch.randn(16, 32, dtype=torch.float32, device="cuda").abs()  # positive
        V = torch.randn(32, 64, dtype=torch.float16, device="cuda")
        alpha = torch.rand(16, dtype=torch.float32, device="cuda")
        out = output_accumulate(O, P, V, alpha)
        expected = O * alpha[:, None] + (P.half() @ V).float()
        assert torch.allclose(out, expected, atol=5e-1), \
            f"Max diff: {(out - expected).abs().max():.4f}"
