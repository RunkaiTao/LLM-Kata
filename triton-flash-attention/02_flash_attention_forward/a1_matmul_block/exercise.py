"""
Exercise a1: Block matrix multiply with tl.dot.

The most fundamental Triton operation in Flash Attention: load a Q block and
a K block from global memory, compute their dot product QK = Q @ K, and
store the result.

In Flash Attention, K is stored in TRANSPOSED layout (HEAD_DIM x SEQ_LEN)
so that Q @ K_transposed gives the (BLOCK_Q x BLOCK_KV) score matrix directly.

Triton concepts introduced:
    - tl.make_block_ptr:  create a pointer to a 2D tile in global memory
    - tl.load:            load a tile from global memory into SRAM (registers)
    - tl.dot:             matrix multiply two tiles in SRAM
    - tl.store:           write a tile from SRAM back to global memory

Math
----
    Q_block:  shape (BLOCK_Q, HEAD_DIM)     — loaded from Q
    K_block:  shape (HEAD_DIM, BLOCK_KV)    — loaded from K (transposed layout)
    QK = Q_block @ K_block                  — shape (BLOCK_Q, BLOCK_KV)

    This is equivalent to: Q[i, :] · K[j, :] for each (i, j) pair,
    giving the raw (unscaled) attention score between query i and key j.

Example (BLOCK_Q=2, HEAD_DIM=3, BLOCK_KV=2)
-------
    Q_block = [[1, 0, 2],      K_block (transposed) = [[1, 0],
               [0, 1, 1]]                               [0, 1],
                                                         [2, 1]]

    QK = [[1*1+0*0+2*2, 1*0+0*1+2*1],   = [[5, 2],
          [0*1+1*0+1*2, 0*0+1*1+1*1]]      [2, 2]]
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# YOUR TASK: Implement matmul_block_kernel
# ---------------------------------------------------------------------------
@triton.jit
def matmul_block_kernel(
    Q_ptr,          # pointer to Q tensor, shape (BLOCK_Q, HEAD_DIM)
    K_ptr,          # pointer to K tensor, shape (HEAD_DIM, BLOCK_KV) [already transposed]
    Out_ptr,        # pointer to output tensor, shape (BLOCK_Q, BLOCK_KV)
    stride_Q_row,   # stride along Q's row dimension
    stride_Q_col,   # stride along Q's column dimension
    stride_K_row,   # stride along K's row dimension (HEAD_DIM axis)
    stride_K_col,   # stride along K's column dimension (BLOCK_KV axis)
    stride_Out_row, # stride along Out's row dimension
    stride_Out_col, # stride along Out's column dimension
    BLOCK_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Load Q and K blocks, compute QK = Q @ K, store the result.

    Steps:
        1. Q_block_ptr = tl.make_block_ptr(
               base=Q_ptr, shape=(BLOCK_Q, HEAD_DIM),
               strides=(stride_Q_row, stride_Q_col),
               offsets=(0, 0),
               block_shape=(BLOCK_Q, HEAD_DIM),
               order=(1, 0))

        2. K_block_ptr = tl.make_block_ptr(
               base=K_ptr, shape=(HEAD_DIM, BLOCK_KV),
               strides=(stride_K_row, stride_K_col),
               offsets=(0, 0),
               block_shape=(HEAD_DIM, BLOCK_KV),
               order=(0, 1))

        3. Out_block_ptr = tl.make_block_ptr(
               base=Out_ptr, shape=(BLOCK_Q, BLOCK_KV),
               strides=(stride_Out_row, stride_Out_col),
               offsets=(0, 0),
               block_shape=(BLOCK_Q, BLOCK_KV),
               order=(1, 0))

        4. Q_block = tl.load(Q_block_ptr)         (load Q tile into SRAM)
        5. K_block = tl.load(K_block_ptr)         (load K tile into SRAM)
        6. QK = tl.dot(Q_block, K_block)          (matrix multiply in SRAM)
        7. tl.store(Out_block_ptr, QK)            (write result to global memory)
    """
    # Step 1: Make block pointer for Q (BLOCK_Q x HEAD_DIM)
    # order=(1, 0) means column-major within the tile: dimension 1 (HEAD_DIM)
    # is contiguous in memory, dimension 0 (BLOCK_Q) is strided
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr, shape=(BLOCK_Q, HEAD_DIM),
        strides=(stride_Q_row, stride_Q_col),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, HEAD_DIM),
        order=(1, 0))

    # Step 2: Make block pointer for K in transposed layout (HEAD_DIM x BLOCK_KV)
    # order=(0, 1) means row-major: dimension 0 (HEAD_DIM) is contiguous
    # This layout lets tl.dot compute Q @ K^T as a standard matmul
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr, shape=(HEAD_DIM, BLOCK_KV),
        strides=(stride_K_row, stride_K_col),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_KV),
        order=(0, 1))

    # Step 3: Make block pointer for output (BLOCK_Q x BLOCK_KV)
    Out_block_ptr = tl.make_block_ptr(
        base=Out_ptr, shape=(BLOCK_Q, BLOCK_KV),
        strides=(stride_Out_row, stride_Out_col),
        offsets=(0, 0),
        block_shape=(BLOCK_Q, BLOCK_KV),
        order=(1, 0))

    # Step 4: Load Q tile from global memory (HBM) into SRAM (registers)
    Q_block = tl.load(Q_block_ptr)

    # Step 5: Load K tile (already transposed) into SRAM
    K_block = tl.load(K_block_ptr)

    # Step 6: Matrix multiply in SRAM — QK[i,j] = dot(Q[i,:], K[:,j])
    # tl.dot accumulates in float32 even when inputs are float16
    QK = tl.dot(Q_block, K_block)

    # Step 7: Write result back to global memory
    tl.store(Out_block_ptr, QK)


# ---------------------------------------------------------------------------
# Python wrapper (used by tests)
# ---------------------------------------------------------------------------
def matmul_block(Q: torch.Tensor, K_t: torch.Tensor) -> torch.Tensor:
    """
    Compute Q @ K_t using the Triton kernel.

    Args:
        Q:   (BLOCK_Q, HEAD_DIM) float16 CUDA tensor.
        K_t: (HEAD_DIM, BLOCK_KV) float16 CUDA tensor (K already transposed).

    Returns:
        Out: (BLOCK_Q, BLOCK_KV) float32 CUDA tensor.
    """
    BLOCK_Q, HEAD_DIM = Q.shape
    HEAD_DIM_K, BLOCK_KV = K_t.shape
    assert HEAD_DIM == HEAD_DIM_K
    Out = torch.empty((BLOCK_Q, BLOCK_KV), dtype=torch.float32, device=Q.device)
    matmul_block_kernel[(1,)](
        Q, K_t, Out,
        Q.stride(0), Q.stride(1),
        K_t.stride(0), K_t.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_Q=BLOCK_Q, HEAD_DIM=HEAD_DIM, BLOCK_KV=BLOCK_KV,
    )
    return Out


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a1_matmul_block/test_exercise.py -v
