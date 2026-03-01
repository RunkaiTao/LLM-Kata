"""
Transformer block: communication (attention) followed by computation (FFN).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import completed exercises: MultiHeadAttention from 02/c, FeedForward from 02/d
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

MultiHeadAttention = load("02_layers", "c_multi_head_attention").MultiHeadAttention
FeedForward = load("02_layers", "d_feed_forward").FeedForward


# ---------------------------------------------------------------------------
# YOUR TASK: Implement Block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        """
        Args:
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            block_size: Maximum sequence length.
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Compute head_size by integer-dividing n_embd by n_head
        # TODO: Create self.sa as a MultiHeadAttention layer with n_embd, n_head, head_size, block_size, dropout
        # TODO: Create self.ffwd as a FeedForward layer with n_embd and dropout
        # TODO: Create self.ln1 as an nn.LayerNorm over n_embd dimensions
        # TODO: Create self.ln2 as an nn.LayerNorm over n_embd dimensions
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).

        Steps:
        1. Apply layer norm (ln1) to x, pass through self-attention (sa), then add residual (x) back
        2. Apply layer norm (ln2) to x, pass through feed-forward (ffwd), then add residual (x) back
        """
        # TODO: Implement the forward pass with residual connections
        raise NotImplementedError("Implement forward")
