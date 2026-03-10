"""
Online (incremental) softmax over blocks.

Standard safe softmax requires two passes: one to find the max, one to compute
the output. Online softmax does it in a single pass by maintaining running
statistics (max and normalization sum) that are updated incrementally as each
block is processed.

This is the core algorithmic idea behind Flash Attention's memory efficiency:
instead of materializing the full N×N attention matrix, we process it in blocks
and accumulate the result using online softmax.

Key concepts:
- Running max m, running sum d — updated block by block
- Correction factor alpha = exp(m_old - m_new) rescales previous accumulation
- After all blocks, normalize using final running sum

Reference: triton-flash-attention/notes/0003 - Online Softmax.pdf
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

safe_softmax = load("01_attention_fundamentals", "a_safe_softmax").safe_softmax


# ---------------------------------------------------------------------------
# YOUR TASK: Implement online_softmax
# ---------------------------------------------------------------------------
def online_softmax(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Compute softmax over a 1D tensor x by processing it in blocks of block_size.

    Args:
        x:          1D float tensor of length N (N must be divisible by block_size).
        block_size: Number of elements to process per block.

    Returns:
        1D tensor of same shape as x containing softmax probabilities.

    Running state (initialize before the loop):
        m: running maximum, initialized to float('-inf')
        d: running normalization sum, initialized to 0.0

    For each block x[lo : lo + block_size]:
        Step 1: block = x[lo : lo + block_size]
        Step 2: m_block = block.max()                       (max of current block)
        Step 3: m_new = torch.maximum(m, m_block)           (new running max, use torch.maximum)
        Step 4: alpha = torch.exp(m - m_new)                (correction factor for old sum)
        Step 5: d = d * alpha + torch.exp(block - m_new).sum()  (update running sum)
        Step 6: m = m_new                                   (update running max)

    After all blocks:
        Step 7: return torch.exp(x - m) / d
    """
    N = x.shape[0]
    assert N % block_size == 0, "N must be divisible by block_size"

    # Initialize running state
    m = torch.tensor(float("-inf"), dtype=x.dtype, device=x.device)
    d = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    # TODO: Loop over blocks and update m and d following the steps above
    # for lo in range(0, N, block_size):
    #     Step 1: block = x[lo : lo + block_size]
    #     Step 2: m_block = ...
    #     Step 3: m_new = ...     (torch.maximum)
    #     Step 4: alpha = ...     (torch.exp(m - m_new))
    #     Step 5: d = ...         (d * alpha + torch.exp(block - m_new).sum())
    #     Step 6: m = m_new

    # Step 7: return torch.exp(x - m) / d
    pass


# Run tests: pytest triton-flash-attention/01_attention_fundamentals/b_online_softmax/test_exercise.py -v
