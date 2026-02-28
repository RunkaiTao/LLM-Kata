"""
Position-wise feed-forward network for the transformer.
"""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward network with ReLU activation.

    Architecture: Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd, n_embd) -> Dropout
    """

    def __init__(self, n_embd: int, dropout: float = 0.0):
        """
        Args:
            n_embd: Embedding dimension (both input and output size).
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Create self.net as nn.Sequential containing:
        #   1. nn.Linear(n_embd, 4 * n_embd)
        #   2. nn.ReLU()
        #   3. nn.Linear(4 * n_embd, n_embd)
        #   4. nn.Dropout(dropout)
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")
