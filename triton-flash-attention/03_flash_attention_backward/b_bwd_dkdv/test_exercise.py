import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

flash_attention_forward = load(
    "02_flash_attention_forward", "b_fwd_kernel"
).flash_attention_forward
compute_delta = load(
    "03_flash_attention_backward", "a_bwd_preprocess"
).compute_delta
from exercise import compute_dk_dv

BATCH = 1
HEADS = 2
SEQ_LEN = 256
HEAD_DIM = 64


def get_reference_gradients(Q, K, V, causal):
    """Compute reference dK and dV via PyTorch autograd."""
    Qf = Q.float().detach().requires_grad_(True)
    Kf = K.float().detach().requires_grad_(True)
    Vf = V.float().detach().requires_grad_(True)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    scores = Qf @ Kf.transpose(-2, -1) * scale
    if causal:
        mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
    import torch.nn.functional as F
    weights = F.softmax(scores, dim=-1)
    out = weights @ Vf
    dO = torch.randn_like(out)
    out.backward(dO)
    return dO.half(), Kf.grad.half(), Vf.grad.half()


@pytest.fixture
def setup():
    torch.manual_seed(42)
    dtype = torch.float16
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    return Q, K, V


class TestBwdDkDv:
    def test_output_shapes(self, setup):
        Q, K, V = setup
        O, M = flash_attention_forward(Q, K, V, causal=False)
        dO = torch.randn_like(O)
        D = compute_delta(O, dO)
        dK, dV = compute_dk_dv(Q, K, V, dO, M, D, causal=False)
        assert dK.shape == K.shape, f"dK shape mismatch: {dK.shape}"
        assert dV.shape == V.shape, f"dV shape mismatch: {dV.shape}"

    def test_dk_allclose_reference(self, setup):
        Q, K, V = setup
        dO, ref_dK, _ = get_reference_gradients(Q, K, V, causal=False)
        O, M = flash_attention_forward(Q, K, V, causal=False)
        D = compute_delta(O, dO)
        dK, _ = compute_dk_dv(Q, K, V, dO, M, D, causal=False)
        assert torch.allclose(dK, ref_dK, atol=1e-2), \
            f"dK max diff: {(dK - ref_dK).abs().max():.4f}"

    def test_dv_allclose_reference(self, setup):
        Q, K, V = setup
        dO, _, ref_dV = get_reference_gradients(Q, K, V, causal=False)
        O, M = flash_attention_forward(Q, K, V, causal=False)
        D = compute_delta(O, dO)
        _, dV = compute_dk_dv(Q, K, V, dO, M, D, causal=False)
        assert torch.allclose(dV, ref_dV, atol=1e-2), \
            f"dV max diff: {(dV - ref_dV).abs().max():.4f}"
