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
        # TODO: Create self.key as a linear projection from n_embd to head_size, no bias (use nn.Linear)
        # TODO: Create self.query as a linear projection from n_embd to head_size, no bias (use nn.Linear)
        # TODO: Create self.value as a linear projection from n_embd to head_size, no bias (use nn.Linear)
        # TODO: Register a buffer named 'tril' containing a lower-triangular matrix of ones,
        #       size block_size x block_size (use register_buffer, torch.tril, torch.ones)
        # TODO: Create self.dropout using nn.Dropout with the given dropout rate
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, head_size).

        Steps:
        1. Extract B, T, C from x.shape
        2. Compute k by passing x through the key projection -> (B, T, head_size)
        3. Compute q by passing x through the query projection -> (B, T, head_size)
        4. Compute attention scores wei as the scaled dot product of q and transposed k;
           scale by 1/sqrt(head_size) -> (B, T, T)
        5. Apply causal mask: fill positions where tril[:T, :T] is 0 with -inf (use masked_fill)
        6. Normalize wei with softmax along the last dimension (use F.softmax)
        7. Apply dropout to wei
        8. Compute v by passing x through the value projection -> (B, T, head_size)
        9. Compute output out as the matrix product of wei and v -> (B, T, head_size)
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v  # (B, T, head_size)
        return out
