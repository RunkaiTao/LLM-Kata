"""
Exercise a2: Apply softmax scaling and causal mask to QK scores in Triton.

After computing QK = Q @ K^T (exercise a1), Flash Attention applies:
  1. Softmax scaling: QK * (1/sqrt(d_k))
  2. Causal masking:  set future positions to -1e6 so softmax maps them to ~0

This exercise teaches two key Triton patterns:
    - tl.arange:  generate index vectors for position-based masking
    - tl.where:   conditional selection (the Triton equivalent of torch.where)

Math
----
    QK_scaled = QK * softmax_scale

    If causal:
        For each (i, j):
            query_pos[i] = query_offset + i
            key_pos[j]   = key_offset + j
            if query_pos[i] < key_pos[j]:
                QK_scaled[i, j] = -1e6      (effectively -inf for softmax)

    mask[i, j] = (query_offset + i) >= (key_offset + j)
    QK_scaled = QK_scaled * softmax_scale + where(mask, 0, -1e6)

    Note: in Flash Attention STAGE==2 (diagonal block), scaling is applied
    BEFORE masking, then both are combined in a single expression:
        QK_block * softmax_scale + tl.where(mask, 0, -1e6)

Example (BLOCK_Q=3, BLOCK_KV=3, query_offset=0, key_offset=0, scale=0.5)
-------
    QK = [[4, 2, 6],         mask = [[T, F, F],     (pos 0 >= 0,1,2)
          [2, 4, 2],                  [T, T, F],     (pos 1 >= 0,1,2)
          [6, 2, 4]]                  [T, T, T]]     (pos 2 >= 0,1,2)

    QK_scaled = [[4*0.5 + 0,    2*0.5 + (-1e6), 6*0.5 + (-1e6)],
                 [2*0.5 + 0,    4*0.5 + 0,       2*0.5 + (-1e6)],
                 [6*0.5 + 0,    2*0.5 + 0,       4*0.5 + 0      ]]
              = [[2.0,   -1e6, -1e6],
                 [1.0,    2.0, -1e6],
                 [3.0,    1.0,  2.0]]
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# YOUR TASK: Implement scale_causal_mask_kernel
# ---------------------------------------------------------------------------
@triton.jit
def scale_causal_mask_kernel(
    QK_ptr,             # pointer to QK scores, shape (BLOCK_Q, BLOCK_KV)
    Out_ptr,            # pointer to output, shape (BLOCK_Q, BLOCK_KV)
    softmax_scale,      # float scalar: 1 / sqrt(HEAD_DIM)
    query_offset,       # int: absolute position of first query in this block
    key_offset,         # int: absolute position of first key in this block
    stride_QK_row,
    stride_QK_col,
    stride_Out_row,
    stride_Out_col,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    CAUSAL: tl.constexpr,   # tl.constexpr bool: whether to apply causal mask
):
    """
    Load QK scores, apply scaling and optional causal mask, store result.

    Steps:
        1. QK_block_ptr = tl.make_block_ptr(
               base=QK_ptr, shape=(BLOCK_Q, BLOCK_KV),
               strides=(stride_QK_row, stride_QK_col),
               offsets=(0, 0), block_shape=(BLOCK_Q, BLOCK_KV), order=(1, 0))

        2. Out_block_ptr = tl.make_block_ptr(
               base=Out_ptr, shape=(BLOCK_Q, BLOCK_KV),
               strides=(stride_Out_row, stride_Out_col),
               offsets=(0, 0), block_shape=(BLOCK_Q, BLOCK_KV), order=(1, 0))

        3. QK_block = tl.load(QK_block_ptr)

        4. if CAUSAL:
               offs_q = query_offset + tl.arange(0, BLOCK_Q)       (query positions)
               offs_kv = key_offset + tl.arange(0, BLOCK_KV)       (key positions)
               mask = offs_q[:, None] >= offs_kv[None, :]           (lower-triangular)
               QK_scaled = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
           else:
               QK_scaled = QK_block * softmax_scale

        5. tl.store(Out_block_ptr, QK_scaled)
    """
    # TODO: Step 1 — make block pointer for QK input
    # TODO: Step 2 — make block pointer for output
    # TODO: Step 3 — load QK_block
    # TODO: Step 4 — apply scaling (and causal mask if CAUSAL)
    # TODO: Step 5 — store result
    pass


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def scale_causal_mask(
    QK: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
    query_offset: int = 0,
    key_offset: int = 0,
) -> torch.Tensor:
    """
    Apply softmax scaling and optional causal mask via Triton kernel.

    Args:
        QK:             (BLOCK_Q, BLOCK_KV) float32 CUDA tensor (raw scores).
        softmax_scale:  1 / sqrt(HEAD_DIM).
        causal:         Whether to apply causal mask.
        query_offset:   Absolute position of first query.
        key_offset:     Absolute position of first key.

    Returns:
        Out: (BLOCK_Q, BLOCK_KV) float32 CUDA tensor (scaled + masked scores).
    """
    BLOCK_Q, BLOCK_KV = QK.shape
    Out = torch.empty_like(QK)
    scale_causal_mask_kernel[(1,)](
        QK, Out, softmax_scale, query_offset, key_offset,
        QK.stride(0), QK.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, CAUSAL=causal,
    )
    return Out


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a2_scale_causal_mask/test_exercise.py -v
