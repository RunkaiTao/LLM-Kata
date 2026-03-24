"""
Tests for _attn_fwd_inner.

We test by building a minimal _attn_fwd wrapper kernel that calls
_attn_fwd_inner and stores its result, then comparing against the
reference_attention output.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

triton = pytest.importorskip("triton")
import triton.language as tl

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention
from exercise import _attn_fwd_inner

BATCH = 1
HEADS = 2
SEQ_LEN = 128
HEAD_DIM = 64
BLOCK_SIZE_Q = 64
BLOCK_SIZE_KV = 64


@triton.autotune(
    [triton.Config({"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
                   num_stages=1, num_warps=4)],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _minimal_fwd(
    Q, K, V, softmax_scale, M, O,
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE, NUM_HEADS: tl.constexpr, SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr, STAGE: tl.constexpr,
):
    block_index_q = tl.program_id(0)
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    qvk_offset = (index_batch.to(tl.int64) * stride_Q_batch
                  + index_head.to(tl.int64) * stride_Q_head)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_SIZE_KV), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), block_shape=(BLOCK_SIZE_KV, HEAD_DIM), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0))

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    Q_block = tl.load(Q_block_ptr)

    if STAGE == 1 or STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            4 - STAGE, offs_q, offs_kv, SEQ_LEN)
    if STAGE == 3:
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block, l_i, m_i, Q_block, K_block_ptr, V_block_ptr,
            block_index_q, softmax_scale, BLOCK_SIZE_Q, BLOCK_SIZE_KV,
            2, offs_q, offs_kv, SEQ_LEN)

    m_i += tl.math.log(l_i)
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def run_fwd(Q, K, V, causal):
    B, H, T, D = Q.shape
    scale = 1.0 / (D ** 0.5)
    O = torch.empty_like(Q)
    M = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    grid = lambda args: (triton.cdiv(T, args["BLOCK_SIZE_Q"]), B * H, 1)
    _minimal_fwd[grid](
        Q, K, V, scale, M, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, T, D, STAGE=stage,
    )
    return O, M


@pytest.fixture
def tensors():
    torch.manual_seed(42)
    dtype = torch.float16
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device="cuda") * 0.5
    return Q, K, V


class TestFwdInner:
    def test_non_causal_output_close_to_reference(self, tensors):
        Q, K, V = tensors
        tri_O, _ = run_fwd(Q, K, V, causal=False)
        ref_O = reference_attention(Q.float(), K.float(), V.float(), causal=False).half()
        assert torch.allclose(tri_O, ref_O, atol=1e-2), \
            f"Non-causal output max diff: {(tri_O - ref_O).abs().max()}"

    def test_causal_output_close_to_reference(self, tensors):
        Q, K, V = tensors
        tri_O, _ = run_fwd(Q, K, V, causal=True)
        ref_O = reference_attention(Q.float(), K.float(), V.float(), causal=True).half()
        assert torch.allclose(tri_O, ref_O, atol=1e-2), \
            f"Causal output max diff: {(tri_O - ref_O).abs().max()}"

    def test_logsumexp_M_finite(self, tensors):
        """The stored logsumexp values M must all be finite (no NaN/Inf)."""
        Q, K, V = tensors
        _, M = run_fwd(Q, K, V, causal=False)
        assert torch.isfinite(M).all(), "M (logsumexp) contains NaN or Inf"

    def test_different_head_dims(self):
        """Should work for HEAD_DIM=32 as well."""
        torch.manual_seed(7)
        dtype = torch.float16
        Q = torch.randn(1, 1, SEQ_LEN, 32, dtype=dtype, device="cuda") * 0.5
        K = torch.randn(1, 1, SEQ_LEN, 32, dtype=dtype, device="cuda") * 0.5
        V = torch.randn(1, 1, SEQ_LEN, 32, dtype=dtype, device="cuda") * 0.5
        ref_O = reference_attention(Q.float(), K.float(), V.float(), causal=False).half()
        tri_O, _ = run_fwd(Q, K, V, causal=False)
        assert torch.allclose(tri_O, ref_O, atol=1e-2)
