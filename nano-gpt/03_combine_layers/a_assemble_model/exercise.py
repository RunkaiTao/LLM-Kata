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
        # TODO: Implement __init__ following the docstring above
        # Step 1: self.block_size = ...   (store block_size)
        # Step 2: self.embeddings = ...   (Embeddings(vocab_size, block_size, n_embd))
        # Step 3: self.blocks = ...       (nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]))
        # Step 4: self.lm_head = ...      (LMHead(n_embd, vocab_size))
        # Step 5: self.apply(self._init_weights)
        pass

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Rules:
        - If module is nn.Linear: set weights to Normal(mean=0.0, std=0.02),
          and if bias exists, set to zeros.
        - If module is nn.Embedding: set weights to Normal(mean=0.0, std=0.02).

        Hint: Use torch.nn.init.normal_() and torch.nn.init.zeros_()
        """
        # TODO: Implement _init_weights following the rules above
        # Step 1: if isinstance(module, nn.Linear): init weights Normal(0, 0.02), bias zeros if exists
        # Step 2: if isinstance(module, nn.Embedding): init weights Normal(0, 0.02)
        pass

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
        # TODO: Implement forward following the steps above
        # Step 1: x = ...       (self.embeddings(idx))
        # Step 2: x = ...       (self.blocks(x))
        # Step 3: logits = ...  (self.lm_head(x))
        # Step 4: if targets is not None: reshape logits to (B*T, C), targets to (B*T), compute loss = F.cross_entropy(...)
        #         else: loss = None
        # Step 5: return (logits, loss)
        pass

# Run tests: pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v
# Test individual parts:
# pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestModelAssembly
# pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestInitWeights
# pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestForwardPass
