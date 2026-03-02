"""
Multi-head attention: multiple self-attention heads in parallel.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import completed exercise: Head from 02/b_self_attention
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

Head = load("02_layers", "b_self_attention").Head


# ---------------------------------------------------------------------------
# YOUR TASK: Implement MultiHeadAttention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            head_size: Output dimension of each head.
            block_size: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Create self.heads as an nn.ModuleList containing n_head Head instances
        #       (each with n_embd, head_size, block_size, dropout)
        # TODO: Create self.proj as a linear layer projecting from head_size * n_head back to n_embd (use nn.Linear)
        # TODO: Create self.dropout using nn.Dropout with the given dropout rate
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).

        Steps:
        1. Run each head on x and concatenate outputs along dim=-1.
        2. Apply the projection linear layer.
        3. Apply dropout.
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")

# Run tests: pytest nano-gpt/02_layers/c_multi_head_attention/test_exercise.py -v
