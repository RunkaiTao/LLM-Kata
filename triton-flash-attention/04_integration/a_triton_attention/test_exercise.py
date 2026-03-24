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
from exercise import TritonAttention

BATCH = 2
HEADS = 4
SEQ_LEN = 256
HEAD_DIM = 64


@pytest.fixture
def tensors():
    torch.manual_seed(42)
    dtype = torch.float16
    Q = (torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
         .requires_grad_(True))
    K = (torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
         .requires_grad_(True))
    V = (torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
         .requires_grad_(True))
    return Q, K, V


def make_tensors_grad(seed=42):
    torch.manual_seed(seed)
    dtype = torch.float16
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda").mul(0.5).requires_grad_(True)
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda").mul(0.5).requires_grad_(True)
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda").mul(0.5).requires_grad_(True)
    return Q, K, V


class TestTritonAttention:
    def test_output_shape(self):
        Q, K, V = make_tensors_grad()
        scale = 1.0 / (HEAD_DIM ** 0.5)
        O = TritonAttention.apply(Q, K, V, False, scale)
        assert O.shape == Q.shape, f"Expected {Q.shape}, got {O.shape}"

    def test_forward_non_causal_allclose_reference(self):
        Q, K, V = make_tensors_grad()
        scale = 1.0 / (HEAD_DIM ** 0.5)
        O = TritonAttention.apply(Q, K, V, False, scale).half()
        ref = reference_attention(Q.detach().float(), K.detach().float(), V.detach().float(),
                                  causal=False).half()
        assert torch.allclose(O, ref, atol=1e-2), \
            f"Non-causal max diff: {(O - ref).abs().max():.4f}"

    def test_forward_causal_allclose_reference(self):
        Q, K, V = make_tensors_grad()
        scale = 1.0 / (HEAD_DIM ** 0.5)
        O = TritonAttention.apply(Q, K, V, True, scale).half()
        ref = reference_attention(Q.detach().float(), K.detach().float(), V.detach().float(),
                                  causal=True).half()
        assert torch.allclose(O, ref, atol=1e-2), \
            f"Causal max diff: {(O - ref).abs().max():.4f}"

    def test_backward_gradients_allclose(self):
        """dQ, dK, dV from TritonAttention.backward must match PyTorch autograd."""
        torch.manual_seed(0)
        dtype = torch.float16

        # Reference via PyTorch autograd
        Qr = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda").mul(0.5).requires_grad_(True)
        Kr = Qr.detach().clone().requires_grad_(True)
        Vr = Qr.detach().clone().requires_grad_(True)
        scale = 1.0 / (HEAD_DIM ** 0.5)
        ref_O = reference_attention(Qr.float(), Kr.float(), Vr.float(), causal=False)
        dO = torch.randn_like(ref_O).half()
        ref_O.half().backward(dO)
        ref_dQ, ref_dK, ref_dV = Qr.grad.clone(), Kr.grad.clone(), Vr.grad.clone()

        # Triton implementation
        Qt = Qr.detach().clone().requires_grad_(True)
        Kt = Kr.detach().clone().requires_grad_(True)
        Vt = Vr.detach().clone().requires_grad_(True)
        tri_O = TritonAttention.apply(Qt, Kt, Vt, False, scale)
        tri_O.backward(dO)

        assert torch.allclose(Qt.grad, ref_dQ, atol=1e-2), f"dQ max diff: {(Qt.grad - ref_dQ).abs().max():.4f}"
        assert torch.allclose(Kt.grad, ref_dK, atol=1e-2), f"dK max diff: {(Kt.grad - ref_dK).abs().max():.4f}"
        assert torch.allclose(Vt.grad, ref_dV, atol=1e-2), f"dV max diff: {(Vt.grad - ref_dV).abs().max():.4f}"

    def test_end_to_end_gradient_flow(self):
        """A simple loss.backward() call must complete without error."""
        Q, K, V = make_tensors_grad()
        scale = 1.0 / (HEAD_DIM ** 0.5)
        O = TritonAttention.apply(Q, K, V, True, scale)
        loss = O.sum()
        loss.backward()  # should not raise
        assert Q.grad is not None
        assert K.grad is not None
        assert V.grad is not None
