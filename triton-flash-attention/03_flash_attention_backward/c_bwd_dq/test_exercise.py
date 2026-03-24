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
from exercise import compute_dq

BATCH = 1
HEADS = 2
SEQ_LEN = 256
HEAD_DIM = 64


def get_reference_dq(Q, K, V, causal):
    """Compute reference dQ via PyTorch autograd."""
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
    return dO.half(), Qf.grad.half()


@pytest.fixture
def setup():
    torch.manual_seed(42)
    dtype = torch.float16
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    return Q, K, V


class TestBwdDq:
    def test_output_shape(self, setup):
        Q, K, V = setup
        O, M = flash_attention_forward(Q, K, V, causal=False)
        dO = torch.randn_like(O)
        D = compute_delta(O, dO)
        dQ = compute_dq(Q, K, V, dO, M, D, causal=False)
        assert dQ.shape == Q.shape, f"dQ shape mismatch: {dQ.shape}"

    def test_dq_allclose_reference_non_causal(self, setup):
        Q, K, V = setup
        dO, ref_dQ = get_reference_dq(Q, K, V, causal=False)
        O, M = flash_attention_forward(Q, K, V, causal=False)
        D = compute_delta(O, dO)
        dQ = compute_dq(Q, K, V, dO, M, D, causal=False)
        assert torch.allclose(dQ, ref_dQ, atol=1e-2), \
            f"dQ max diff: {(dQ - ref_dQ).abs().max():.4f}"

    def test_dq_allclose_reference_causal(self, setup):
        Q, K, V = setup
        dO, ref_dQ = get_reference_dq(Q, K, V, causal=True)
        O, M = flash_attention_forward(Q, K, V, causal=True)
        D = compute_delta(O, dO)
        dQ = compute_dq(Q, K, V, dO, M, D, causal=True)
        assert torch.allclose(dQ, ref_dQ, atol=1e-2), \
            f"Causal dQ max diff: {(dQ - ref_dQ).abs().max():.4f}"
