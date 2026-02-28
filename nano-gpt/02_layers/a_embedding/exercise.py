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
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: Token indices of shape (B, T), where B=batch, T=sequence length.

        Returns:
            Tensor of shape (B, T, n_embd): sum of token and position embeddings.

        Steps:
        1. Look up token embeddings: self.token_embedding_table(idx) -> (B, T, C)
        2. Create position indices: torch.arange(T, device=idx.device) -> (T,)
        3. Look up position embeddings: self.position_embedding_table(positions) -> (T, C)
        4. Sum them: tok_emb + pos_emb -> (B, T, C) via broadcasting
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        return tok_emb + pos_emb
