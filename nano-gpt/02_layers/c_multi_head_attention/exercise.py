"""
Multi-head attention: multiple self-attention heads in parallel.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# PROVIDED: Single attention head (do not modify)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


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
