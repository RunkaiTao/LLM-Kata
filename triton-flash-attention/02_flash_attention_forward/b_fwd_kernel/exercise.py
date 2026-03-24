"""
Flash Attention full forward kernel: _attn_fwd.

This Triton kernel orchestrates the full forward pass of Flash Attention 2.
Each thread block handles one tile of queries (BLOCK_SIZE_Q rows) for one
(batch, head) pair. It:
1. Sets up block pointers for Q, K (transposed), V, and O
2. Initializes running statistics m_i, l_i, and the output accumulator O_block
3. Calls _attn_fwd_inner for the non-masked region (and diagonal for causal)
4. Finalizes the output (normalize by l_i, store logsumexp M for backward)

The @triton.autotune decorator automatically searches for the best block sizes
and hardware parameters (num_warps, num_stages) for your GPU.

K is accessed in TRANSPOSED layout (shape HEAD_DIM x SEQ_LEN) so that
QK^T = Q @ K_transposed is a standard matrix multiplication in Triton.

Reference: triton-flash-attention/triton/flash_attention.py lines 82-246
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

_attn_fwd_inner = load(
    "02_flash_attention_forward", "a_fwd_inner"
)._attn_fwd_inner

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention


# ---------------------------------------------------------------------------
# YOUR TASK: Implement _attn_fwd
# ---------------------------------------------------------------------------
@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64, 128]
        for BLOCK_SIZE_KV in [32, 64]
        for num_stages in [3, 4, 7]
        for num_warps in [2, 4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    softmax_scale,
    M,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN) — logsumexp output
    O,           # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS:   tl.constexpr,
    SEQ_LEN:     tl.constexpr,
    HEAD_DIM:    tl.constexpr,
    BLOCK_SIZE_Q:  tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE:       tl.constexpr,
):
    """
    Full forward kernel. One program per (Q-block, batch*head) pair.

    Steps:
    1.  block_index_q = tl.program_id(0)
    2.  index_batch_head = tl.program_id(1)
    3.  index_batch = index_batch_head // NUM_HEADS
    4.  index_head  = index_batch_head % NUM_HEADS
    5.  qvk_offset = (index_batch.to(tl.int64) * stride_Q_batch
                      + index_head.to(tl.int64) * stride_Q_head)
    6.  Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_Q_seq, stride_Q_dim),
            offsets=(block_index_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0))
    7.  K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset, shape=(HEAD_DIM, SEQ_LEN),
            strides=(stride_K_dim, stride_K_seq),   # <-- transposed strides
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_SIZE_KV), order=(0, 1))
    8.  V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_V_seq, stride_V_dim),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_KV, HEAD_DIM), order=(1, 0))
    9.  O_block_ptr = tl.make_block_ptr(
            base=O + qvk_offset, shape=(SEQ_LEN, HEAD_DIM),
            strides=(stride_O_seq, stride_O_dim),
            offsets=(block_index_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, HEAD_DIM), order=(1, 0))
    10. offs_q  = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    11. offs_kv = tl.arange(0, BLOCK_SIZE_KV)
    12. m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    13. l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    14. O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    15. Q_block = tl.load(Q_block_ptr)

    --- Process non-masked region (and non-causal) ---
    16. if STAGE == 1 or STAGE == 3:
            O_block, l_i, m_i = _attn_fwd_inner(
                O_block, l_i, m_i, Q_block,
                K_block_ptr, V_block_ptr,
                block_index_q, softmax_scale,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV,
                4 - STAGE,      # maps STAGE 3 -> 1, STAGE 1 -> 3
                offs_q, offs_kv, SEQ_LEN)

    --- Process causal diagonal block ---
    17. if STAGE == 3:
            O_block, l_i, m_i = _attn_fwd_inner(
                O_block, l_i, m_i, Q_block,
                K_block_ptr, V_block_ptr,
                block_index_q, softmax_scale,
                BLOCK_SIZE_Q, BLOCK_SIZE_KV,
                2,              # diagonal stage
                offs_q, offs_kv, SEQ_LEN)

    --- Epilogue ---
    18. m_i += tl.math.log(l_i)           (convert to true logsumexp for backward)
    19. O_block = O_block / l_i[:, None]  (normalize output)
    20. tl.store(M + index_batch_head * SEQ_LEN + offs_q, m_i)
    21. tl.store(O_block_ptr, O_block.to(O.type.element_ty))
    """
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # TODO: Step 1:  block_index_q = tl.program_id(0)
    # TODO: Step 2:  index_batch_head = tl.program_id(1)
    # TODO: Step 3:  index_batch = index_batch_head // NUM_HEADS
    # TODO: Step 4:  index_head = index_batch_head % NUM_HEADS
    # TODO: Step 5:  qvk_offset = ...
    # TODO: Step 6:  Q_block_ptr = tl.make_block_ptr(...)
    # TODO: Step 7:  K_block_ptr = tl.make_block_ptr(...)   (transposed strides!)
    # TODO: Step 8:  V_block_ptr = tl.make_block_ptr(...)
    # TODO: Step 9:  O_block_ptr = tl.make_block_ptr(...)
    # TODO: Step 10: offs_q  = ...
    # TODO: Step 11: offs_kv = ...
    # TODO: Step 12: m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # TODO: Step 13: l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # TODO: Step 14: O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    # TODO: Step 15: Q_block = tl.load(Q_block_ptr)
    # TODO: Steps 16-17: call _attn_fwd_inner (once or twice depending on STAGE)
    # TODO: Steps 18-21: epilogue (normalize, store M and O_block)
    pass


# ---------------------------------------------------------------------------
# Helper: run _attn_fwd and return (O, M)
# ---------------------------------------------------------------------------
def flash_attention_forward(Q, K, V, causal=False):
    """
    Run _attn_fwd kernel on (B, H, T, D) tensors.

    Args:
        Q, K, V: float16 CUDA tensors of shape (B, H, T, D).
        causal:  whether to apply causal masking.

    Returns:
        O: output tensor of same shape as Q.
        M: logsumexp tensor of shape (B, H, T).
    """
    B, H, T, D = Q.shape
    softmax_scale = 1.0 / (D ** 0.5)
    O = torch.empty_like(Q)
    M = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
    stage = 3 if causal else 1
    grid = lambda args: (triton.cdiv(T, args["BLOCK_SIZE_Q"]), B * H, 1)
    _attn_fwd[grid](
        Q=Q, K=K, V=V, softmax_scale=softmax_scale, M=M, O=O,
        stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),   stride_Q_dim=Q.stride(3),
        stride_K_batch=K.stride(0), stride_K_head=K.stride(1),
        stride_K_seq=K.stride(2),   stride_K_dim=K.stride(3),
        stride_V_batch=V.stride(0), stride_V_head=V.stride(1),
        stride_V_seq=V.stride(2),   stride_V_dim=V.stride(3),
        stride_O_batch=O.stride(0), stride_O_head=O.stride(1),
        stride_O_seq=O.stride(2),   stride_O_dim=O.stride(3),
        BATCH_SIZE=B, NUM_HEADS=H, SEQ_LEN=T, HEAD_DIM=D,
        STAGE=stage,
    )
    return O, M


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/b_fwd_kernel/test_exercise.py -v
