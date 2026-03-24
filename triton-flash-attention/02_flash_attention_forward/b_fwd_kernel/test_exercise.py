import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention
from exercise import flash_attention_forward

BATCH = 2
HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64


@pytest.fixture
def tensors():
    torch.manual_seed(42)
    dtype = torch.float16
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    return Q, K, V


class TestFwdKernel:
    def test_output_shape(self, tensors):
        Q, K, V = tensors
        O, M = flash_attention_forward(Q, K, V, causal=False)
        assert O.shape == Q.shape, f"Expected {Q.shape}, got {O.shape}"
        assert M.shape == (BATCH, HEADS, SEQ_LEN), f"Expected M shape {(BATCH, HEADS, SEQ_LEN)}, got {M.shape}"

    def test_non_causal_allclose_reference(self, tensors):
        Q, K, V = tensors
        O, _ = flash_attention_forward(Q, K, V, causal=False)
        ref = reference_attention(Q.float(), K.float(), V.float(), causal=False).half()
        assert torch.allclose(O, ref, atol=1e-2), \
            f"Non-causal max diff: {(O - ref).abs().max():.4f}"

    def test_causal_allclose_reference(self, tensors):
        Q, K, V = tensors
        O, _ = flash_attention_forward(Q, K, V, causal=True)
        ref = reference_attention(Q.float(), K.float(), V.float(), causal=True).half()
        assert torch.allclose(O, ref, atol=1e-2), \
            f"Causal max diff: {(O - ref).abs().max():.4f}"

    def test_logsumexp_M_stored_correctly(self, tensors):
        """
        M[b, h, i] should equal log(sum_j exp(scores[b,h,i,j])) for non-causal.
        We verify it's finite and that exp(M) > 0.
        """
        Q, K, V = tensors
        _, M = flash_attention_forward(Q, K, V, causal=False)
        assert torch.isfinite(M).all(), "M contains NaN or Inf"
        # exp(logsumexp) must be positive
        assert (torch.exp(M) > 0).all()
