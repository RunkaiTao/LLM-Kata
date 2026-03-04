"""
Token and position embedding layer for nano-GPT.
"""
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """
    Combines token embeddings and position embeddings.

    Token embedding: maps each token ID to a vector of size n_embd.
    Position embedding: maps each position (0..block_size-1) to a vector of size n_embd.
    The forward pass looks up both and sums them.
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            block_size: Maximum sequence length.
            n_embd: Embedding dimension.
        """
        super().__init__()
        # TODO: Implement __init__ following the docstring above
        # Step 1: self.token_embedding_table = ...     (nn.Embedding: vocab_size -> n_embd)
        # Step 2: self.position_embedding_table = ...  (nn.Embedding: block_size -> n_embd)
        pass

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token indices of shape (B, T), where B=batch, T=sequence length.

        Returns:
            Tensor of shape (B, T, n_embd): sum of token and position embeddings.

        Steps:
        1. Look up tok_emb by passing idx through the token embedding table -> (B, T, C)
        2. Create position indices as a range from 0 to T on the same device as idx (use torch.arange) -> (T,)
        3. Look up pos_emb by passing position indices through the position embedding table -> (T, C)
        4. Return the sum of tok_emb and pos_emb (broadcasting aligns shapes) -> (B, T, C)
        """
        # TODO: Implement forward following the steps above
        # Step 1: tok_emb = ...  (self.token_embedding_table(idx))
        # Step 2: pos = ...      (torch.arange(T, device=idx.device))
        # Step 3: pos_emb = ...  (self.position_embedding_table(pos))
        # Step 4: return ...     (tok_emb + pos_emb)
        pass

# Run tests: pytest nano-gpt/02_layers/a_embedding/test_exercise.py -v
