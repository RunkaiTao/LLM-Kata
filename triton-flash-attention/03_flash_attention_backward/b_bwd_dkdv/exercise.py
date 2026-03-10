"""
Flash Attention backward kernel: dK and dV — _attn_bwd_dk_dv.

This kernel fixes one block of K and V (kept in SRAM) and iterates over
all Q blocks to accumulate dK and dV.

The key gradient formulas are:

    dV[j] = sum_i  P[i,j]^T  @ dO[i]         (P^T weighted sum of dO)
    dK[j] = sum_i  dS[i,j]^T @ Q[i]  * scale  (scaled dS^T times Q)

where:
    P[i,j] = exp(QK^T[i,j] - M[i])            (attention weight, recomputed from logsumexp M)
    dP[i,j] = dO[i] @ V[j]^T                  (gradient of output w.r.t attention weights)
    dS[i,j] = P[i,j] * (dP[i,j] - D[i])       (pre-softmax gradient)
    D[i] = rowsum(O[i] * dO[i])               (delta from backward preprocess)

The outer loop iterates over Q blocks while K and V remain fixed, which is
the opposite of the forward pass (which fixes Q and iterates KV).

Reference: triton-flash-attention/triton/flash_attention.py lines 384-513
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
# YOUR TASK: Implement _attn_bwd_dk_dv
# ---------------------------------------------------------------------------
@triton.jit
def _attn_bwd_dk_dv(
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
    Compute dK and dV for one KV block by iterating over all Q blocks.

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
    start_kv = index_block_kv * BLOCK_KV
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)

    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)

    Load K and V blocks (these stay in SRAM for the whole inner loop):
      K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)
      V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim)

    offs_q = tl.arange(0, BLOCK_Q)
    qT_ptrs = Q  + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim  # transposed
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim

    --- Inner loop (iterate over Q blocks) ---
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        Step 1:  qT_block = tl.load(qT_ptrs)                          (Q^T block: HEAD_DIM x BLOCK_Q)
        Step 2:  offs_q = curr_q + tl.arange(0, BLOCK_Q)
                 m = tl.load(M + offs_q)                               (logsumexp for current Q block)
        Step 3:  QK_T_block = softmax_scale * tl.dot(K_block, qT_block)  (KQ^T = P^T before softmax)
        Step 4:  P_T_block = tl.math.exp(QK_T_block - m[None, :])     (P^T, recomputed)
        Step 5:  if STAGE == 3:
                     mask_block = offs_q[None, :] >= offs_kv[:, None]  (causal mask)
                     P_T_block = tl.where(mask_block, P_T_block, 0.0)
        Step 6:  dO_block = tl.load(dO_ptrs)
        Step 7:  dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)   (dV += P^T @ dO)
        Step 8:  Di = tl.load(D + offs_q)
        Step 9:  dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)  (dP^T = V @ dO^T)
        Step 10: dS_T_block = P_T_block * (dpT_block - Di[None, :])    (dS^T = P^T * (dP^T - D^T))
        Step 11: dK_block += softmax_scale * tl.dot(dS_T_block.to(tl.float16), tl.trans(qT_block))
        Step 12: curr_q += BLOCK_Q
                 qT_ptrs += BLOCK_Q * stride_seq
                 dO_ptrs += BLOCK_Q * stride_seq

    --- Store results ---
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dV_block_ptrs, dV_block)
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)
    """
    # TODO: Setup — index batch/head, offsets, load K/V blocks, initialize dV/dK accumulators
    # TODO: Inner loop — for each Q block: recompute P^T, accumulate dV and dK
    # TODO: Store dV_block and dK_block
    pass


# ---------------------------------------------------------------------------
# Helper: run dK, dV backward kernel
# ---------------------------------------------------------------------------
def compute_dk_dv(Q, K, V, dO, M, D, causal=False):
    """
    Compute dK and dV using the _attn_bwd_dk_dv kernel.

    Args:
        Q, K, V:  float16 CUDA tensors (B, H, T, D).
        dO:       float16 CUDA tensor  (B, H, T, D) — gradient of output.
        M:        float32 CUDA tensor  (B, H, T)    — logsumexp from forward.
        D:        float32 CUDA tensor  (B, H, T)    — delta from preprocess.
        causal:   whether causal masking was used.

    Returns:
        dK, dV: float32 CUDA tensors of shape (B, H, T, D).
    """
    B, H, T, Dh = Q.shape
    dK = torch.zeros_like(Q)
    dV = torch.zeros_like(Q)
    dQ_placeholder = torch.zeros_like(Q)

    NUM_WARPS, NUM_STAGES = 4, 3
    BLOCK_Q, BLOCK_KV = 32, 128
    stage = 3 if causal else 1

    grid = (T // BLOCK_KV, 1, B * H)
    _attn_bwd_dk_dv[grid](
        Q=Q, K=K, V=V,
        softmax_scale=1.0 / (Dh ** 0.5),
        dO=dO, dQ=dQ_placeholder, dK=dK, dV=dV,
        M=M, D=D,
        stride_batch=Q.stride(0), stride_head=Q.stride(1),
        stride_seq=Q.stride(2), stride_dim=Q.stride(3),
        NUM_HEADS=H, SEQ_LEN=T,
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, HEAD_DIM=Dh,
        STAGE=stage,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return dK, dV


# Run tests: pytest triton-flash-attention/03_flash_attention_backward/b_bwd_dkdv/test_exercise.py -v
