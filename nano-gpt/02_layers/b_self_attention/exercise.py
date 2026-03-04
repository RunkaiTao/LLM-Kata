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
        # TODO: Implement __init__ following the docstring above
        # Step 1: self.key = ...      (nn.Linear: n_embd -> head_size, bias=False)
        # Step 2: self.query = ...    (nn.Linear: n_embd -> head_size, bias=False)
        # Step 3: self.value = ...    (nn.Linear: n_embd -> head_size, bias=False)
        # Step 4: self.register_buffer('tril', ...)  (torch.tril(torch.ones(block_size, block_size)))
        # Step 5: self.dropout = ...  (nn.Dropout(dropout))
        pass

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
        # TODO: Implement forward following the steps above
        # Step 1: B, T, C = ...  (x.shape)
        # Step 2: k = ...        (self.key(x))
        # Step 3: q = ...        (self.query(x))
        # Step 4: wei = ...      (q @ k.transpose(-2,-1) * C**-0.5)
        # Step 5: wei = ...      (wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')))
        # Step 6: wei = ...      (F.softmax(wei, dim=-1))
        # Step 7: wei = ...      (self.dropout(wei))
        # Step 8: v = ...        (self.value(x))
        # Step 9: out = ...      (wei @ v)
        # return out
        pass

# Run tests: pytest nano-gpt/02_layers/b_self_attention/test_exercise.py -v
