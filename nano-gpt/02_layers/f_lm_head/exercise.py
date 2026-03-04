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
        # TODO: Implement __init__ following the docstring above
        # Step 1: self.ln_f = ...   (nn.LayerNorm(n_embd))
        # Step 2: self.proj = ...   (nn.Linear: n_embd -> vocab_size)
        pass

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
        # TODO: Implement forward following the steps above
        # Step 1: x = ...       (self.ln_f(x))
        # Step 2: logits = ...  (self.proj(x))
        # Step 3: return logits
        pass

# Run tests: pytest nano-gpt/02_layers/f_lm_head/test_exercise.py -v
