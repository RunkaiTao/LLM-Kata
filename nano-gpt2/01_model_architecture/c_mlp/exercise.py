"""
Feed-forward network (MLP) for GPT-2 with GELU activation.

Unlike the simpler nano-gpt kata which uses ReLU, GPT-2 uses GELU with
the tanh approximation. The MLP expands the embedding dimension by 4x,
applies GELU, then projects back to the original dimension.

The c_proj layer is flagged with NANOGPT_SCALE_INIT for scaled weight
initialization in the full model (to prevent residual stream growth).

Reference: Karpathy's build-nanogpt train_gpt2.py lines 42-55
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import completed exercise: GPTConfig
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig instance with n_embd.

        Steps:
        1. Create self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
           — expansion layer (4x wider)
        2. Create self.gelu = nn.GELU(approximate='tanh')
           — GPT-2 uses the tanh approximation of GELU, not exact GELU or ReLU
        3. Create self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
           — projection back to embedding dimension
        4. Set self.c_proj.NANOGPT_SCALE_INIT = 1
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. x = self.c_fc(x)    -> (B, T, 4*C)
        2. x = self.gelu(x)    -> (B, T, 4*C)
        3. x = self.c_proj(x)  -> (B, T, C)
        4. Return x
        """
        # TODO: Implement forward following the steps above
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
