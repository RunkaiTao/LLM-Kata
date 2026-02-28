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
        # TODO: Create self.net as an nn.Sequential containing:
        #   1. A linear layer expanding from n_embd to 4 * n_embd (use nn.Linear)
        #   2. A ReLU activation (use nn.ReLU)
        #   3. A linear layer projecting from 4 * n_embd back to n_embd (use nn.Linear)
        #   4. A dropout layer (use nn.Dropout)
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
