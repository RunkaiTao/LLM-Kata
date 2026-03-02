"""
Language model head: final layer norm followed by linear projection to vocabulary.
"""
import torch
import torch.nn as nn


class LMHead(nn.Module):
    """
    Maps transformer hidden states to vocabulary logits.

    Architecture: LayerNorm(n_embd) -> Linear(n_embd, vocab_size)

    The LayerNorm stabilizes the final hidden states before projecting
    them to unnormalized log-probabilities (logits) over the vocabulary.
    """

    def __init__(self, n_embd: int, vocab_size: int):
        """
        Args:
            n_embd: Embedding dimension (input size).
            vocab_size: Number of tokens in the vocabulary (output size).
        """
        super().__init__()
        # TODO: Create self.ln_f as an nn.LayerNorm over n_embd dimensions
        # TODO: Create self.proj as an nn.Linear mapping n_embd to vocab_size
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states of shape (B, T, n_embd).

        Returns:
            Logits of shape (B, T, vocab_size).

        Steps:
        1. Apply layer norm to x -> (B, T, n_embd)
        2. Project through the linear layer -> (B, T, vocab_size)
        3. Return the logits
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")

# Run tests: pytest nano-gpt/02_layers/f_lm_head/test_exercise.py -v
