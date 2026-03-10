"""
Numerically stable (safe) softmax.

Naive softmax computes exp(x) / sum(exp(x)). For large values of x (e.g. 1000),
exp(x) overflows to inf, producing NaN. The fix is to subtract the row maximum
before exponentiating — this shifts values to <= 0 without changing the output
(the constant cancels in numerator and denominator).

This is the foundational numerical trick used throughout Flash Attention.

Key concepts:
- Subtract row-wise max before exp to prevent overflow
- Result is mathematically identical to naive softmax

Reference: triton-flash-attention/notes/0002 - (Safe) Softmax.pdf
"""
import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# YOUR TASK: Implement safe_softmax
# ---------------------------------------------------------------------------
def safe_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable softmax over the last dimension of x.

    Args:
        x: Input tensor of any shape (..., N).

    Returns:
        Tensor of same shape as x, with values in (0, 1) summing to 1
        along the last dimension.

    Steps:
    1. Compute the row-wise maximum along the last dim (use .max(dim=-1, keepdim=True))
    2. Subtract the maximum from x to get x_shifted
    3. Exponentiate x_shifted (use torch.exp)
    4. Compute the row-wise sum of exp_x along the last dim (use .sum(dim=-1, keepdim=True))
    5. Return exp_x / sum_exp
    """
    # Step 1: m = ...          (row-wise max, keepdim=True)
    # Step 2: x_shifted = ...  (x minus m)
    # Step 3: exp_x = ...      (torch.exp of x_shifted)
    # Step 4: sum_exp = ...    (row-wise sum of exp_x, keepdim=True)
    # Step 5: return exp_x / sum_exp
    pass


# Run tests: pytest triton-flash-attention/01_attention_fundamentals/a_safe_softmax/test_exercise.py -v
