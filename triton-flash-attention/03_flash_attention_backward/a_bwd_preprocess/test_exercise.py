import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import compute_delta

BATCH = 2
HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64


@pytest.fixture
def o_do():
    torch.manual_seed(42)
    dtype = torch.float16
    O  = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda")
    dO = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda")
    return O, dO


class TestBwdPreprocess:
    def test_output_shape(self, o_do):
        O, dO = o_do
        D = compute_delta(O, dO)
        assert D.shape == (BATCH, HEADS, SEQ_LEN), \
            f"Expected shape {(BATCH, HEADS, SEQ_LEN)}, got {D.shape}"

    def test_values_correct(self, o_do):
        """D[b, h, i] must equal the dot product of O[b,h,i] and dO[b,h,i]."""
        O, dO = o_do
        D = compute_delta(O, dO)
        expected = (O.float() * dO.float()).sum(dim=-1)  # (B, H, T)
        assert torch.allclose(D, expected, atol=1e-2), \
            f"Max diff: {(D - expected).abs().max():.4f}"

    def test_matches_pytorch_rowsum(self, o_do):
        """Result must match the simple PyTorch computation (O * dO).sum(-1)."""
        O, dO = o_do
        D = compute_delta(O, dO)
        ref = (O.float() * dO.float()).sum(-1)
        assert torch.allclose(D, ref, atol=1e-2)
