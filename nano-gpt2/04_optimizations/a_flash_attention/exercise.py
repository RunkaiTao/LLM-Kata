"""
Flash Attention: replacing manual attention with F.scaled_dot_product_attention.

In the original CausalSelfAttention (01/b), we manually computed:
  att = (q @ k.T) / sqrt(d) -> masked_fill -> softmax -> att @ v

PyTorch provides F.scaled_dot_product_attention which fuses these operations
into a single, memory-efficient kernel ("flash attention"). This is much faster
on GPU and uses O(T) memory instead of O(T^2) for the attention matrix.

Key changes from 01/b:
- REMOVE: the register_buffer('bias', ...) causal mask — no longer needed
- REPLACE: the manual attention computation with a single function call

Reference: Karpathy's build-nanogpt train_gpt2.py line 36
           (commit 7ee630c: "Switch to flash attention")
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Import completed exercise: GPTConfig
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement CausalSelfAttention using F.scaled_dot_product_attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig instance with n_embd, n_head, block_size.

        Steps (same as 01/b EXCEPT no causal mask buffer):
        1. Assert that n_embd is divisible by n_head
        2. Create self.c_attn — single linear projection from n_embd to 3*n_embd (use nn.Linear)
        3. Create self.c_proj — output projection from n_embd to n_embd (use nn.Linear)
        4. Flag c_proj for scaled initialization (set NANOGPT_SCALE_INIT attribute to 1)
        5. Store n_head and n_embd from config
        6. NO register_buffer needed! Flash attention handles causal masking internally.
        """
        super().__init__()
        # TODO: Implement __init__ (note: NO causal mask buffer needed)
        # Step 1: assert ...
        # Step 2: self.c_attn = ...  (nn.Linear: n_embd -> 3 * n_embd)
        # Step 3: self.c_proj = ...  (nn.Linear: n_embd -> n_embd)
        # Step 4: self.c_proj.NANOGPT_SCALE_INIT = ...
        # Step 5: self.n_head = ..., self.n_embd = ...
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. Unpack B, T, C from x
        2. Project x through c_attn to get combined qkv -> (B, T, 3*C)
        3. Split into q, k, v along dim 2, each of size n_embd (use .split)
        4. Reshape q, k, v for multi-head: (B, n_head, T, head_size) (use .view and .transpose)
        5. Compute attention using flash attention with causal masking
           (use F.scaled_dot_product_attention with is_causal=True)
           — this single call replaces: scale, mask, softmax, matmul
        6. Reassemble heads back to (B, T, C) (use .transpose, .contiguous, .view)
        7. Project through c_proj and return
        """
        # TODO: Implement forward using F.scaled_dot_product_attention
        # Step 1: B, T, C = ...
        # Step 2: qkv = ...                (project x through c_attn)
        # Step 3: q, k, v = ...            (split qkv along dim=2, each size n_embd)
        # Step 4: k = ..., q = ..., v = ... (reshape each to (B, n_head, T, head_size))
        # Step 5: y = ...                  (F.scaled_dot_product_attention with is_causal=True)
        # Step 6: y = ...                  (reassemble heads: .transpose, .contiguous, .view -> (B, T, C))
        # Step 7: y = ...                  (project through c_proj)
        # return y
        pass

# Run tests: pytest nano-gpt2/04_optimizations/a_flash_attention/test_exercise.py -v
