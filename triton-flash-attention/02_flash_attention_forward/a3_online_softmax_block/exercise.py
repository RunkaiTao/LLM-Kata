"""
Exercise a3: Online softmax update step in Triton.

Given a block of (already scaled+masked) QK scores, update the running softmax
statistics m_i (per-row max) and l_i (per-row normalization sum), and produce
the unnormalized probability block P_block.

This is the numerical heart of Flash Attention — the same algorithm from
exercise 01/b_online_softmax, now operating per-row on 2D blocks in Triton.

Triton concepts introduced:
    - tl.max(tensor, axis):    row-wise maximum reduction
    - tl.sum(tensor, axis):    row-wise sum reduction
    - tl.math.exp:             element-wise exponential (Triton's fast exp)
    - tl.maximum:              element-wise max of two tensors

Math
----
Given:
    QK_scaled:  (BLOCK_Q, BLOCK_KV) — scaled (and masked) attention scores
    m_i:        (BLOCK_Q,)          — running max from previous KV blocks
    l_i:        (BLOCK_Q,)          — running normalization sum

Compute:
    m_ij    = maximum(m_i, max_per_row(QK_scaled))   # updated running max
    alpha   = exp(m_i - m_ij)                         # correction factor (≤ 1)
    P_block = exp(QK_scaled - m_ij[:, None])          # shifted scores
    l_ij    = sum_per_row(P_block)                    # row sums of new block
    l_i     = l_i * alpha + l_ij                      # corrected running sum

Why alpha ≤ 1:  m_ij ≥ m_i  →  m_i - m_ij ≤ 0  →  exp(...) ≤ 1

Example (BLOCK_Q=2, BLOCK_KV=2)
-------
    QK_scaled = [[2.0, 1.0],     m_i = [-inf, -inf],   l_i = [0, 0]
                 [0.0, 3.0]]     (first block)

    m_ij  = max([-inf,-inf], [2, 3]) = [2.0, 3.0]
    alpha = exp([-inf-2, -inf-3])    = [0.0, 0.0]
    P     = exp([[0, -1], [-3, 0]])  = [[1.0, 0.368], [0.050, 1.0]]
    l_ij  = [1.368, 1.050]
    l_new = [0*0+1.368, 0*0+1.050]  = [1.368, 1.050]
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# YOUR TASK: Implement online_softmax_block_kernel
# ---------------------------------------------------------------------------
@triton.jit
def online_softmax_block_kernel(
    QK_ptr,         # (BLOCK_Q, BLOCK_KV) scaled scores input
    M_in_ptr,       # (BLOCK_Q,) current running max input
    L_in_ptr,       # (BLOCK_Q,) current running sum input
    M_out_ptr,      # (BLOCK_Q,) updated running max output
    L_out_ptr,      # (BLOCK_Q,) updated running sum output
    P_ptr,          # (BLOCK_Q, BLOCK_KV) P_block output
    Alpha_ptr,      # (BLOCK_Q,) alpha output (correction factor)
    stride_QK_row,
    stride_QK_col,
    stride_P_row,
    stride_P_col,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Compute one online softmax update step.

    Steps:
        1. Load QK_block from QK_ptr using tl.make_block_ptr
        2. Load m_i from M_in_ptr:
               offs = tl.arange(0, BLOCK_Q)
               m_i = tl.load(M_in_ptr + offs)
        3. Load l_i from L_in_ptr:
               l_i = tl.load(L_in_ptr + offs)
        4. m_ij = tl.maximum(m_i, tl.max(QK_block, 1))        (new running max)
        5. alpha = tl.math.exp(m_i - m_ij)                     (correction factor)
        6. QK_shifted = QK_block - m_ij[:, None]               (shift scores)
        7. P_block = tl.math.exp(QK_shifted)                   (unnormalized probs)
        8. l_ij = tl.sum(P_block, 1)                           (row sums)
        9. l_new = l_i * alpha + l_ij                          (updated running sum)
        10. Store m_ij to M_out_ptr, l_new to L_out_ptr,
            P_block to P_ptr, alpha to Alpha_ptr
    """
    # TODO: Steps 1-3: Load inputs
    # TODO: Step 4:    m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
    # TODO: Step 5:    alpha = tl.math.exp(m_i - m_ij)
    # TODO: Steps 6-7: P_block = tl.math.exp(QK_block - m_ij[:, None])
    # TODO: Step 8:    l_ij = tl.sum(P_block, 1)
    # TODO: Step 9:    l_new = l_i * alpha + l_ij
    # TODO: Step 10:   Store all outputs
    pass


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def online_softmax_block(
    QK_scaled: torch.Tensor,
    m_i: torch.Tensor,
    l_i: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one online softmax update via Triton kernel.

    Args:
        QK_scaled: (BLOCK_Q, BLOCK_KV) float32 CUDA tensor.
        m_i:       (BLOCK_Q,) float32 CUDA tensor (running max).
        l_i:       (BLOCK_Q,) float32 CUDA tensor (running sum).

    Returns:
        m_new, l_new, P_block, alpha
    """
    BLOCK_Q, BLOCK_KV = QK_scaled.shape
    m_out = torch.empty_like(m_i)
    l_out = torch.empty_like(l_i)
    P = torch.empty_like(QK_scaled)
    alpha = torch.empty_like(m_i)
    online_softmax_block_kernel[(1,)](
        QK_scaled, m_i, l_i, m_out, l_out, P, alpha,
        QK_scaled.stride(0), QK_scaled.stride(1),
        P.stride(0), P.stride(1),
        BLOCK_Q=BLOCK_Q, BLOCK_KV=BLOCK_KV,
    )
    return m_out, l_out, P, alpha


# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a3_online_softmax_block/test_exercise.py -v
