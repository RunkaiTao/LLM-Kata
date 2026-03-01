"""
Assembling the full GPT language model and implementing the forward pass.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Import completed exercises: Embeddings, Block, LMHead from 02_layers
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

Embeddings = load("02_layers", "a_embedding").Embeddings
Block = load("02_layers", "e_transformer_block").Block
LMHead = load("02_layers", "f_lm_head").LMHead


# ---------------------------------------------------------------------------
# YOUR TASK: Implement GPTLanguageModel.__init__, _init_weights, and forward
# ---------------------------------------------------------------------------
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.0):
        """
        Args:
            vocab_size: Number of tokens in the vocabulary.
            block_size: Maximum sequence length.
            n_embd: Embedding dimension.
            n_head: Number of attention heads.
            n_layer: Number of transformer blocks.
            dropout: Dropout rate.
        """
        super().__init__()
        # TODO: Store block_size as self.block_size for later use
        # TODO: Create self.embeddings as an Embeddings layer (vocab_size, block_size, n_embd)
        # TODO: Create self.blocks as an nn.Sequential of n_layer Block instances
        #       (each with n_embd, n_head, block_size, dropout)
        # TODO: Create self.lm_head as an LMHead layer (n_embd, vocab_size)
        # TODO: Apply weight initialization by calling self.apply with self._init_weights
        raise NotImplementedError("Implement __init__")

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Rules:
        - If module is nn.Linear: set weights to Normal(mean=0.0, std=0.02),
          and if bias exists, set to zeros.
        - If module is nn.Embedding: set weights to Normal(mean=0.0, std=0.02).

        Hint: Use torch.nn.init.normal_() and torch.nn.init.zeros_()
        """
        # TODO: Implement weight initialization
        raise NotImplementedError("Implement _init_weights")

    def forward(self, idx, targets=None):
        """
        Args:
            idx: Token indices of shape (B, T).
            targets: Target token indices of shape (B, T), or None.

        Returns:
            logits: Tensor of shape (B, T, vocab_size) if targets is None,
                    or (B*T, vocab_size) if targets provided.
            loss: Scalar cross-entropy loss, or None if no targets.

        Steps:
        1. Pass idx through self.embeddings to get combined embeddings -> (B, T, C)
        2. Pass x through the transformer blocks -> (B, T, C)
        3. Pass x through self.lm_head to get logits -> (B, T, vocab_size)
        4. If targets is not None:
              Reshape logits to (B*T, vocab_size) and targets to (B*T),
              then compute loss using F.cross_entropy
           Else:
              loss = None
        5. Return (logits, loss)
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Implement forward")
