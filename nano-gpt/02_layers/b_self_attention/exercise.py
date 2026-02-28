"""
Single head of causal self-attention.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float = 0.0):
        """
        Args:
            n_embd: Input embedding dimension.
            head_size: Output dimension of this attention head.
            block_size: Maximum sequence length (for the causal mask).
            dropout: Dropout rate for attention weights.
        """
        super().__init__()
        # TODO: Create self.key   = nn.Linear(n_embd, head_size, bias=False)
        # TODO: Create self.query = nn.Linear(n_embd, head_size, bias=False)
        # TODO: Create self.value = nn.Linear(n_embd, head_size, bias=False)
        # TODO: Register a buffer 'tril' = torch.tril(torch.ones(block_size, block_size))
        # TODO: Create self.dropout = nn.Dropout(dropout)
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, head_size).

        Steps:
        1. Extract B, T, C from x.shape
        2. Compute k = self.key(x)    -> (B, T, head_size)
        3. Compute q = self.query(x)  -> (B, T, head_size)
        4. Compute attention scores: wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        5. Apply causal mask: wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        6. Apply softmax: wei = F.softmax(wei, dim=-1)
        7. Apply dropout: wei = self.dropout(wei)
        8. Compute v = self.value(x)  -> (B, T, head_size)
        9. Compute output: out = wei @ v -> (B, T, head_size)
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")
