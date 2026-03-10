"""
Flash Attention backward preprocessing: _attn_bwd_preprocess.

Before computing gradients for Q, K, and V, we precompute delta values D where:

    D[b, h, i] = sum_d( O[b, h, i, d] * dO[b, h, i, d] )
               = rowsum(O * dO)   (element-wise product, summed over head_dim)

This is the "Di" term that appears in the gradient formula:
    dS[i, j] = P[i, j] * (dP[i, j] - Di)

Precomputing D into a separate buffer avoids redundant computation
in the backward kernels for dQ, dK, dV.

Reference: triton-flash-attention/triton/flash_attention.py lines 249-280
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

flash_attention_forward = load(
    "02_flash_attention_forward", "b_fwd_kernel"
).flash_attention_forward


# ---------------------------------------------------------------------------
# YOUR TASK: Implement _attn_bwd_preprocess
# ---------------------------------------------------------------------------
@triton.jit
def _attn_bwd_preprocess(
    O,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) — forward output
    dO,          # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM) — output gradient
    D,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN) — delta output to compute
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM:     tl.constexpr,
):
    """
    Compute D[b, h, i] = rowsum(O[b, h, i, :] * dO[b, h, i, :]).

    Each program handles BLOCK_SIZE_Q rows for one (batch, head) pair.

    Steps:
    1. block_index_q = tl.program_id(0)
    2. offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    3. index_batch_head = tl.program_id(1)
    4. offs_dim = tl.arange(0, HEAD_DIM)
    5. O_block = tl.load(
           O + index_batch_head * HEAD_DIM * SEQ_LEN
             + offs_q[:, None] * HEAD_DIM
             + offs_dim[None, :])
       -> shape (BLOCK_SIZE_Q, HEAD_DIM)
    6. dO_block = tl.load(
           dO + index_batch_head * HEAD_DIM * SEQ_LEN
              + offs_q[:, None] * HEAD_DIM
              + offs_dim[None, :]).to(tl.float32)
    7. D_block = tl.sum(dO_block * O_block, axis=1)
       -> shape (BLOCK_SIZE_Q,)
    8. tl.store(D + index_batch_head * SEQ_LEN + offs_q, D_block)
    """
    # TODO: Step 1: block_index_q = tl.program_id(0)
    # TODO: Step 2: offs_q = ...
    # TODO: Step 3: index_batch_head = tl.program_id(1)
    # TODO: Step 4: offs_dim = tl.arange(0, HEAD_DIM)
    # TODO: Step 5: O_block = tl.load(O + index_batch_head * HEAD_DIM * SEQ_LEN
    #                                   + offs_q[:, None] * HEAD_DIM
    #                                   + offs_dim[None, :])
    # TODO: Step 6: dO_block = tl.load(...).to(tl.float32)
    # TODO: Step 7: D_block = tl.sum(dO_block * O_block, axis=1)
    # TODO: Step 8: tl.store(D + index_batch_head * SEQ_LEN + offs_q, D_block)
    pass


# ---------------------------------------------------------------------------
# Helper: run preprocess kernel, return D tensor
# ---------------------------------------------------------------------------
def compute_delta(O: torch.Tensor, dO: torch.Tensor) -> torch.Tensor:
    """
    Compute D = rowsum(O * dO).

    Args:
        O:  float16 CUDA tensor of shape (B, H, T, D).
        dO: float16 CUDA tensor of shape (B, H, T, D).

    Returns:
        D: float32 CUDA tensor of shape (B, H, T).
    """
    B, H, T, D = O.shape
    BLOCK_SIZE_Q = 128
    D_out = torch.empty((B, H, T), device=O.device, dtype=torch.float32)
    grid = (T // BLOCK_SIZE_Q, B * H)
    _attn_bwd_preprocess[grid](
        O=O, dO=dO, D=D_out,
        SEQ_LEN=T,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        HEAD_DIM=D,
    )
    return D_out


# Run tests: pytest triton-flash-attention/03_flash_attention_backward/a_bwd_preprocess/test_exercise.py -v
