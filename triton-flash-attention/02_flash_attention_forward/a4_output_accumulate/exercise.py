"""
Exercise a4: Output accumulation with rescaling in Triton.

After computing the online softmax update (exercise a3), we need to:
  1. Rescale the old output O_block by alpha (correction for updated max)
  2. Accumulate the new contribution P_block @ V_block

This teaches the crucial Triton pattern of using tl.dot with an accumulator
argument — the third argument to tl.dot adds the result to an existing tensor
instead of creating a new one: tl.dot(A, B, acc) computes acc + A @ B.

Triton concepts introduced:
    - tl.dot with 3 arguments:  fused multiply-add (acc + A @ B)
    - Broadcasting alpha[:, None] to rescale a 2D block
    - .to(tl.float16):  cast before tl.dot (which needs matching input types)

Math
----
    O_block_new = O_block_old * alpha[:, None]     +  P_block @ V_block
                  ^^^ rescale old accumulator           ^^^ new contribution

    In Triton this becomes:
        O_block = O_block * alpha[:, None]            # rescale in-place
        P_block = P_block.to(tl.float16)              # cast for tl.dot
        O_block = tl.dot(P_block, V_block, O_block)   # fused: O += P @ V

    Why cast P_block to float16?
    tl.dot requires both input matrices to have the same dtype (float16).
    The accumulator O_block stays in float32 for numerical precision.

Example (BLOCK_Q=1, BLOCK_KV=2, HEAD_DIM=2)
-------
    O_old = [[10, 20]]     alpha = [0.5]
    P = [[0.7, 0.3]]       V = [[1, 0],
                                 [0, 1]]

    O_rescaled = [[10, 20]] * 0.5     = [[5, 10]]
    PV         = [[0.7, 0.3]] @ V     = [[0.7, 0.3]]
    O_new      = [[5, 10]] + [[0.7, 0.3]] = [[5.7, 10.3]]
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# YOUR TASK: Implement output_accumulate_kernel
# ---------------------------------------------------------------------------
@triton.jit
def output_accumulate_kernel(
    O_ptr,          # (BLOCK_Q, HEAD_DIM) old output accumulator (float32)
    P_ptr,          # (BLOCK_Q, BLOCK_KV) attention weights (float32, will cast to fp16)
    V_ptr,          # (BLOCK_KV, HEAD_DIM) value block (float16)
    Alpha_ptr,      # (BLOCK_Q,) correction factor (float32)
    Out_ptr,        # (BLOCK_Q, HEAD_DIM) new output (float32)
    stride_O_row, stride_O_col,
    stride_P_row, stride_P_col,
    stride_V_row, stride_V_col,
    stride_Out_row, stride_Out_col,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Rescale old output by alpha, accumulate P @ V, store result.

    Steps:
        1. Load O_block from O_ptr using tl.make_block_ptr
           (shape=(BLOCK_Q, HEAD_DIM), block_shape=(BLOCK_Q, HEAD_DIM))

        2. Load P_block from P_ptr using tl.make_block_ptr
           (shape=(BLOCK_Q, BLOCK_KV), block_shape=(BLOCK_Q, BLOCK_KV))

        3. Load V_block from V_ptr using tl.make_block_ptr
           (shape=(BLOCK_KV, HEAD_DIM), block_shape=(BLOCK_KV, HEAD_DIM))

        4. Load alpha:
               offs = tl.arange(0, BLOCK_Q)
               alpha = tl.load(Alpha_ptr + offs)

        5. O_block = O_block * alpha[:, None]          (rescale old output)

        6. P_block = P_block.to(tl.float16)            (cast for tl.dot)

        7. O_block = tl.dot(P_block, V_block, O_block) (fused: O += P @ V)

        8. Store O_block to Out_ptr using tl.make_block_ptr
    """
    # TODO: Steps 1-3: Load O_block, P_block, V_block
    # TODO: Step 4:    Load alpha
    # TODO: Step 5:    O_block = O_block * alpha[:, None]
    # TODO: Step 6:    P_block = P_block.to(tl.float16)
    # TODO: Step 7:    O_block = tl.dot(P_block, V_block, O_block)
    # TODO: Step 8:    Store O_block
    pass


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def output_accumulate(
    O_block: torch.Tensor,
    P_block: torch.Tensor,
    V_block: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Rescale and accumulate output via Triton kernel.

    Args:
        O_block: (BLOCK_Q, HEAD_DIM) float32 CUDA tensor.
        P_block: (BLOCK_Q, BLOCK_KV) float32 CUDA tensor.
        V_block: (BLOCK_KV, HEAD_DIM) float16 CUDA tensor.
        alpha:   (BLOCK_Q,) float32 CUDA tensor.

    Returns:
        O_new: (BLOCK_Q, HEAD_DIM) float32 CUDA tensor.
    """
    BLOCK_Q, HEAD_DIM = O_block.shape
    _, BLOCK_KV = P_block.shape
    Out = torch.empty_like(O_block)
    output_accumulate_kernel[(1,)](
        O_block, P_block, V_block, alpha, Out,
        O_block.stride(0), O_block.stride(1),
        P_block.stride(0), P_block.stride(1),
        V_block.stride(0), V_block.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV, HEAD_DIM=HEAD_DIM,
    )
    return Out


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a4_output_accumulate/test_exercise.py -v
