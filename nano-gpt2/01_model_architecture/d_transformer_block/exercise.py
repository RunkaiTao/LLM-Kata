"""
Transformer block with pre-LayerNorm and residual connections.

Each block applies:
  x = x + Attention(LayerNorm(x))   # self-attention with residual
  x = x + MLP(LayerNorm(x))         # feed-forward with residual

This is the "pre-norm" architecture used by GPT-2, where LayerNorm is applied
BEFORE the sub-layer (not after, as in the original Transformer paper).

Reference: Karpathy's build-nanogpt train_gpt2.py lines 57-69
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import completed exercises: GPTConfig, CausalSelfAttention, MLP
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
CausalSelfAttention = load("01_model_architecture", "b_causal_self_attention").CausalSelfAttention
MLP = load("01_model_architecture", "c_mlp").MLP


# ---------------------------------------------------------------------------
# YOUR TASK: Implement Block
# ---------------------------------------------------------------------------
class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig instance.

        Steps:
        1. Create self.ln_1 — layer norm over n_embd (use nn.LayerNorm)
        2. Create self.attn — causal self-attention sub-layer (use CausalSelfAttention)
        3. Create self.ln_2 — layer norm over n_embd (use nn.LayerNorm)
        4. Create self.mlp — feed-forward sub-layer (use MLP)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.ln_1 = ...  (nn.LayerNorm over n_embd)
        # Step 2: self.attn = ...  (CausalSelfAttention)
        # Step 3: self.ln_2 = ...  (nn.LayerNorm over n_embd)
        # Step 4: self.mlp = ...   (MLP)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. Apply layer norm (ln_1) to x, pass through self-attention (attn),
           then add residual connection back to x
        2. Apply layer norm (ln_2) to x, pass through feed-forward (mlp),
           then add residual connection back to x
        3. Return x
        """
        # TODO: Implement forward following the steps above
        # Step 1: x = x + ...  (apply ln_1, then attn, then add residual)
        # Step 2: x = x + ...  (apply ln_2, then mlp, then add residual)
        # return x
        pass

# Run tests: pytest nano-gpt2/01_model_architecture/d_transformer_block/test_exercise.py -v
