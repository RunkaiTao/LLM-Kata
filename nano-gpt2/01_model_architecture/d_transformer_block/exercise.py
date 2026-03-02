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
        1. Create self.ln_1 = nn.LayerNorm(config.n_embd)  — norm before attention
        2. Create self.attn = CausalSelfAttention(config)   — multi-head attention
        3. Create self.ln_2 = nn.LayerNorm(config.n_embd)  — norm before MLP
        4. Create self.mlp = MLP(config)                     — feed-forward network
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. x = x + self.attn(self.ln_1(x))   — pre-norm attention + residual
        2. x = x + self.mlp(self.ln_2(x))    — pre-norm MLP + residual
        3. Return x
        """
        # TODO: Implement forward following the steps above
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
