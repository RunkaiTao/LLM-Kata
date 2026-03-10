"""
Flash Attention backward kernel: dQ — _attn_bwd_dq.

This kernel fixes one block of Q (kept in SRAM) and iterates over all K and V
blocks to accumulate dQ.

The key gradient formula is:

    dQ[i] = sum_j  dS[i,j] @ K[j]  * scale

where:
    P[i,j]  = exp(QK^T[i,j] - M[i])        (attention weight, recomputed from M)
    dP[i,j] = dO[i] @ V[j]^T               (gradient w.r.t attention weights)
    dS[i,j] = P[i,j] * (dP[i,j] - D[i])   (pre-softmax gradient)
    D[i]    = rowsum(O[i] * dO[i])         (delta from backward preprocess)

The structure mirrors _attn_bwd_dk_dv but in the opposite direction:
  - dK,dV: fix KV, iterate Q  (done in previous exercise)
  - dQ:    fix Q,  iterate KV (this exercise)

Reference: triton-flash-attention/triton/flash_attention.py lines 283-381
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention
flash_attention_forward = load(
    "02_flash_attention_forward", "b_fwd_kernel"
).flash_attention_forward
compute_delta = load(
    "03_flash_attention_backward", "a_bwd_preprocess"
).compute_delta


# ---------------------------------------------------------------------------
# YOUR TASK: Implement _attn_bwd_dq
# ---------------------------------------------------------------------------
@triton.jit
def _attn_bwd_dq(
    Q, K, V,
    softmax_scale,
    dO,
    dQ, dK, dV,
    M,          # (BATCH_SIZE, NUM_HEADS, SEQ_LEN) — logsumexp from forward
    D,          # (BATCH_SIZE, NUM_HEADS, SEQ_LEN) — delta from preprocess
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q:   tl.constexpr,
    BLOCK_KV:  tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    STAGE:     tl.constexpr,
):
    """
    Compute dQ for one Q block by iterating over all KV blocks.

    --- Setup ---
    index_batch_head = tl.program_id(2)
    index_batch = index_batch_head // NUM_HEADS
    index_head  = index_batch_head % NUM_HEADS
    offset_batch_head     = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    offset_batch_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64)

    Offset all tensor pointers by offset_batch_head:
      Q += offset_batch_head; K += offset_batch_head; V += offset_batch_head
      dO += offset_batch_head; dQ += offset_batch_head
      dK += offset_batch_head; dV += offset_batch_head
    Offset M and D by offset_batch_head_seq:
      M += offset_batch_head_seq; D += offset_batch_head_seq

    offs_dim = tl.arange(0, HEAD_DIM)
    index_block_kv = tl.program_id(0)
    start_q = index_block_kv * BLOCK_Q       (note: grid dim 0 indexes Q blocks here)
    offs_q = start_q + tl.arange(0, BLOCK_Q)

    Load Q block and dO block (stay in SRAM):
      Q_block  = tl.load(Q  + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
      dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
      dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
      M_block  = tl.load(M + offs_q)[:, None]
      Di       = tl.load(D + offs_q)

    offs_kv = tl.arange(0, BLOCK_KV)
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim  # K transposed
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim  # V transposed

    --- Inner loop (iterate over KV blocks) ---
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    for blk_idx in range(num_steps):
        Step 1:  K_T_block = tl.load(kT_ptrs)                          (K^T: HEAD_DIM x BLOCK_KV)
        Step 2:  V_T_block = tl.load(vT_ptrs)
        Step 3:  QK_block = softmax_scale * tl.dot(Q_block, K_T_block) (Q @ K^T)
        Step 4:  P_block = tl.math.exp(QK_block - M_block)             (P, recomputed)
        Step 5:  if STAGE == 3:
                     offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
                     mask_block = offs_q[:, None] >= offs_kv[None, :]   (causal mask)
                     P_block = tl.where(mask_block, P_block, 0.0)
        Step 6:  dP_block = tl.dot(dO_block, V_T_block).to(tl.float32) (dP = dO @ V^T)
        Step 7:  dS_block = P_block * (dP_block - Di[:, None])          (dS = P * (dP - D))
        Step 8:  dQ_block += softmax_scale * tl.dot(dS_block.to(tl.float16), tl.trans(K_T_block))
        Step 9:  curr_kv += BLOCK_KV
                 kT_ptrs += BLOCK_KV * stride_seq
                 vT_ptrs += BLOCK_KV * stride_seq

    --- Store result ---
    dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dQ_block_ptrs, dQ_block)
    """
    # TODO: Setup — index batch/head, offsets, load Q/dO blocks, init dQ accumulator
    # TODO: Inner loop — for each KV block: recompute P, compute dS, accumulate dQ
    # TODO: Store dQ_block
    pass


# ---------------------------------------------------------------------------
# Helper: run dQ backward kernel
# ---------------------------------------------------------------------------
def compute_dq(Q, K, V, dO, M, D, causal=False):
    """
    Compute dQ using the _attn_bwd_dq kernel.

    Args:
        Q, K, V:  float16 CUDA tensors (B, H, T, D).
        dO:       float16 CUDA tensor  (B, H, T, D).
        M:        float32 CUDA tensor  (B, H, T)    — logsumexp.
        D:        float32 CUDA tensor  (B, H, T)    — delta.
        causal:   whether causal masking was used.

    Returns:
        dQ: float32 CUDA tensor of shape (B, H, T, D).
    """
    B, H, T, Dh = Q.shape
    dQ = torch.zeros_like(Q)
    dK_placeholder = torch.zeros_like(Q)
    dV_placeholder = torch.zeros_like(Q)

    NUM_WARPS, NUM_STAGES = 4, 3
    BLOCK_Q, BLOCK_KV = 128, 32
    stage = 3 if causal else 1

    grid = (T // BLOCK_Q, 1, B * H)
    _attn_bwd_dq[grid](
        Q=Q, K=K, V=V,
        softmax_scale=1.0 / (Dh ** 0.5),
        dO=dO, dQ=dQ, dK=dK_placeholder, dV=dV_placeholder,
        M=M, D=D,
        stride_batch=Q.stride(0), stride_head=Q.stride(1),
        stride_seq=Q.stride(2), stride_dim=Q.stride(3),
        NUM_HEADS=H, SEQ_LEN=T,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, HEAD_DIM=Dh,
        STAGE=stage,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return dQ


# Run tests: pytest triton-flash-attention/03_flash_attention_backward/c_bwd_dq/test_exercise.py -v
