"""
Batched multi-head causal self-attention for GPT-2.

Unlike the simpler nano-gpt kata (which uses separate Head + MultiHeadAttention
with nn.ModuleList), GPT-2 computes all attention heads in a single batched
operation through one linear layer (c_attn). This is more efficient and matches
the real GPT-2 architecture.

Key concepts:
- Single c_attn linear layer projects to 3 * n_embd (Q, K, V concatenated)
- Reshape to (B, n_head, T, head_size) for parallel head computation
- Manual scaled dot-product attention with causal masking
- Output projection c_proj with NANOGPT_SCALE_INIT flag for scaled initialization

Reference: Karpathy's build-nanogpt train_gpt2.py lines 12-40
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
# YOUR TASK: Implement CausalSelfAttention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        """
        Args:
            config: GPTConfig instance with n_embd, n_head, block_size.

        Steps:
        1. Assert that n_embd is divisible by n_head
        2. Create self.c_attn — a single linear projection from n_embd to 3*n_embd
           that computes Q, K, V for all heads in one shot (use nn.Linear)
        3. Create self.c_proj — output projection from n_embd to n_embd (use nn.Linear)
        4. Flag c_proj for scaled initialization by setting NANOGPT_SCALE_INIT attribute to 1
        5. Store n_head and n_embd from config as instance attributes
        6. Register a buffer named 'bias' — a lower-triangular causal mask
           of shape (1, 1, block_size, block_size)
           (use register_buffer, torch.tril, torch.ones)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: assert ...
        # Step 2: self.c_attn = ...  (nn.Linear: n_embd -> 3 * n_embd)
        # Step 3: self.c_proj = ...  (nn.Linear: n_embd -> n_embd)
        # Step 4: self.c_proj.NANOGPT_SCALE_INIT = ...
        # Step 5: self.n_head = ..., self.n_embd = ...
        # Step 6: self.register_buffer("bias", ...)  (lower-triangular mask, shape (1, 1, block_size, block_size))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. Unpack batch size B, sequence length T, and embedding dim C from x
        2. Project x through c_attn to get combined qkv -> (B, T, 3*C)
        3. Split qkv into q, k, v along dim 2, each of size n_embd (use .split)
        4. Reshape q, k, v for multi-head: split C into n_head groups of head_size,
           then transpose to (B, n_head, T, head_size) (use .view and .transpose)
        5. Compute scaled dot-product attention scores between q and k,
           scaling by 1/sqrt(head_size) -> (B, n_head, T, T) (use math.sqrt)
        6. Apply causal mask from self.bias (sliced to T), filling masked positions
           with -inf (use .masked_fill)
        7. Normalize attention weights to probabilities (use F.softmax over last dim)
        8. Apply attention weights to v -> (B, n_head, T, head_size)
        9. Reassemble all heads: transpose and reshape back to (B, T, C)
           (use .transpose, .contiguous, .view)
        10. Project through c_proj and return
        """
        # TODO: Implement forward following the steps above
        # Step 1:  B, T, C = ...
        # Step 2:  qkv = ...                (project x through c_attn)
        # Step 3:  q, k, v = ...            (split qkv along dim=2, each size n_embd)
        # Step 4:  k = ..., q = ..., v = ... (reshape each to (B, n_head, T, head_size) via .view and .transpose)
        # Step 5:  att = ...                (scaled dot-product: (q @ k^T) / sqrt(head_size))
        # Step 6:  att = ...                (apply causal mask with .masked_fill, fill with -inf)
        # Step 7:  att = ...                (softmax over last dim)
        # Step 8:  y = ...                  (att @ v)
        # Step 9:  y = ...                  (reassemble heads: .transpose, .contiguous, .view back to (B, T, C))
        # Step 10: y = ...                  (project through c_proj)
        # return y
        pass

# Run tests: pytest nano-gpt2/01_model_architecture/b_causal_self_attention/test_exercise.py -v
