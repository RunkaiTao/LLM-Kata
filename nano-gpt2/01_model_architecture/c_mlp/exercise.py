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
        1. Create self.c_fc — expansion linear layer from n_embd to 4*n_embd (use nn.Linear)
        2. Create self.gelu — GELU activation with tanh approximation
           (use nn.GELU with approximate param; GPT-2 uses tanh, not exact GELU)
        3. Create self.c_proj — projection linear layer from 4*n_embd back to n_embd (use nn.Linear)
        4. Flag c_proj for scaled initialization (set NANOGPT_SCALE_INIT attribute to 1)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.c_fc = ...    (nn.Linear: n_embd -> 4 * n_embd)
        # Step 2: self.gelu = ...    (nn.GELU with approximate="tanh")
        # Step 3: self.c_proj = ...  (nn.Linear: 4 * n_embd -> n_embd)
        # Step 4: self.c_proj.NANOGPT_SCALE_INIT = ...
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. Pass x through the expansion layer c_fc -> (B, T, 4*C)
        2. Apply GELU activation -> (B, T, 4*C)
        3. Pass through projection layer c_proj -> (B, T, C)
        4. Return the result
        """
        # TODO: Implement forward following the steps above
        # Step 1: x = ...  (pass through c_fc)
        # Step 2: x = ...  (apply gelu activation)
        # Step 3: x = ...  (pass through c_proj)
        # return x
        pass

# Run tests: pytest nano-gpt2/01_model_architecture/c_mlp/test_exercise.py -v
