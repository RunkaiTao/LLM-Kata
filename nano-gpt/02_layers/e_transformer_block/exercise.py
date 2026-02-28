"""
Transformer block: communication (attention) followed by computation (FFN).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# PROVIDED: Head, MultiHeadAttention, FeedForward (do not modify)
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


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd: int, n_head: int, head_size: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


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
