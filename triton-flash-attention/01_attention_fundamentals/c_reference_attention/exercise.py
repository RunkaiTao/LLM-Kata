"""
Standard (reference) scaled dot-product attention.

This implements the full O(N²) attention computation that Flash Attention
optimizes. It serves as the correctness baseline: all Flash Attention
implementations are validated by comparing their output against this function.

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Key concepts:
- Scale scores by 1/sqrt(head_dim) to keep variance stable
- Optional causal (autoregressive) masking: position i can only attend to j <= i
- Softmax over the sequence dimension (last axis of scores)

Reference: triton-flash-attention/notes/0001 - Multi Head Attention.pdf
           triton-flash-attention/triton/flash_attention.py test_op() lines 690-700
"""
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# YOUR TASK: Implement reference_attention
# ---------------------------------------------------------------------------
def reference_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
    softmax_scale: float = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).
        K: Key tensor of shape   (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).
        V: Value tensor of shape  (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).
        causal:        If True, apply causal (lower-triangular) mask.
        softmax_scale: Scale factor. Defaults to 1 / sqrt(HEAD_DIM).

    Returns:
        Output tensor of shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM).

    Steps:
    1. head_dim = Q.shape[-1]
    2. softmax_scale = 1.0 / math.sqrt(head_dim)   (only if the argument is None)
    3. scores = Q @ K.transpose(-2, -1) * softmax_scale
       -> shape (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
    4. If causal:
         mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device))
         scores = scores.masked_fill(mask == 0, float("-inf"))
    5. weights = F.softmax(scores, dim=-1)
    6. out = weights @ V
    7. return out
    """
    # Step 1: head_dim = ...
    # Step 2: if softmax_scale is None: softmax_scale = ...
    # Step 3: scores = ...          (Q @ K.transpose(-2, -1) * softmax_scale)
    # Step 4: if causal:
    #             SEQ_LEN = Q.shape[-2]
    #             mask = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=Q.device))
    #             scores = scores.masked_fill(mask == 0, float("-inf"))
    # Step 5: weights = ...         (F.softmax(scores, dim=-1))
    # Step 6: out = ...             (weights @ V)
    # Step 7: return out
    pass


# Run tests: pytest triton-flash-attention/01_attention_fundamentals/c_reference_attention/test_exercise.py -v
