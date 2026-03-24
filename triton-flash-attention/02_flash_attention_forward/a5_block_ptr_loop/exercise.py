"""
Exercise a5: Block pointer loop with tl.advance.

Flash Attention iterates over KV blocks using block pointers. After processing
each block, the pointer is advanced to the next block with tl.advance().

This exercise teaches the iteration pattern in isolation: loop over column
blocks of a matrix, loading each block and computing a simple reduction
(row-wise max), to practice tl.advance without the complexity of the full
attention computation.

Triton concepts introduced:
    - tl.advance(block_ptr, (row_offset, col_offset)):  move a block pointer
    - tl.multiple_of(val, N):  compiler hint that val is divisible by N
    - Looping with range(lo, hi, BLOCK_SIZE) in Triton

The pattern in _attn_fwd_inner:
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))       # skip to starting block
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)   # compiler hint
        K_block = tl.load(K_block_ptr)
        ...process block...
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))   # next block
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))

This exercise simplifies it: loop over column blocks of a (ROWS x COLS) matrix,
loading each (ROWS x BLOCK_SIZE) block and tracking the row-wise max.

Example (ROWS=2, COLS=8, BLOCK_SIZE=4)
-------
    Matrix = [[1, 5, 3, 2,  |  4, 0, 6, 1],
              [3, 1, 0, 4,  |  2, 7, 1, 3]]
              ^^^ block 0 ^^^   ^^^ block 1 ^^^

    Block 0: row_max = [5, 4]
    Block 1: row_max = max([5,4], [6,7]) = [6, 7]
    Final output: [6, 7]
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# YOUR TASK: Implement block_ptr_loop_kernel
# ---------------------------------------------------------------------------
@triton.jit
def block_ptr_loop_kernel(
    X_ptr,           # pointer to input matrix (ROWS x COLS)
    Out_ptr,         # pointer to output vector (ROWS,) — row-wise max
    stride_X_row,
    stride_X_col,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Loop over column blocks, track row-wise max, store result.

    Steps:
        1. X_block_ptr = tl.make_block_ptr(
               base=X_ptr, shape=(ROWS, COLS),
               strides=(stride_X_row, stride_X_col),
               offsets=(0, 0),
               block_shape=(ROWS, BLOCK_SIZE),
               order=(1, 0))

        2. row_max = tl.zeros([ROWS], dtype=tl.float32) - float("inf")

        3. for start in range(0, COLS, BLOCK_SIZE):
               start = tl.multiple_of(start, BLOCK_SIZE)     # compiler hint

               a. block = tl.load(X_block_ptr)               # (ROWS, BLOCK_SIZE)
               b. block_max = tl.max(block, 1)               # (ROWS,) max per row
               c. row_max = tl.maximum(row_max, block_max)   # update running max
               d. X_block_ptr = tl.advance(X_block_ptr, (0, BLOCK_SIZE))  # next block

        4. offs = tl.arange(0, ROWS)
           tl.store(Out_ptr + offs, row_max)
    """
    # TODO: Step 1 — make block pointer for X
    # TODO: Step 2 — initialize row_max to -inf
    # TODO: Step 3 — loop: load, reduce, advance
    # TODO: Step 4 — store row_max
    pass


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def block_ptr_loop(X: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Compute row-wise max by looping over column blocks via Triton.

    Args:
        X:          (ROWS, COLS) float32 CUDA tensor. COLS must be divisible by block_size.
        block_size: Number of columns per block.

    Returns:
        row_max: (ROWS,) float32 CUDA tensor.
    """
    ROWS, COLS = X.shape
    assert COLS % block_size == 0
    Out = torch.empty(ROWS, dtype=torch.float32, device=X.device)
    block_ptr_loop_kernel[(1,)](
        X, Out,
        X.stride(0), X.stride(1),
        ROWS=ROWS, COLS=COLS, BLOCK_SIZE=block_size,
    )
    return Out


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a5_block_ptr_loop/test_exercise.py -v
