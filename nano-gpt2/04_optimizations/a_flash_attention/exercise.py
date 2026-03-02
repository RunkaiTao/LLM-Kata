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

        Steps (same as 01/b EXCEPT step 6):
        1. Assert config.n_embd % config.n_head == 0
        2. Create self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        3. Create self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        4. Set self.c_proj.NANOGPT_SCALE_INIT = 1
        5. Store self.n_head and self.n_embd
        6. NO register_buffer needed! Flash attention handles causal masking internally.
        """
        super().__init__()
        # TODO: Implement __init__ (note: NO causal mask buffer needed)
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. B, T, C = x.size()
        2. qkv = self.c_attn(x) -> (B, T, 3*C)
        3. Split: q, k, v = qkv.split(self.n_embd, dim=2) -> each (B, T, C)
        4. Reshape k: k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) -> (B, nh, T, hs)
        5. Same for q and v
        6. FLASH ATTENTION: y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
           — this single call replaces: scale, mask, softmax, matmul
        7. Reassemble: y = y.transpose(1, 2).contiguous().view(B, T, C)
        8. Output projection: y = self.c_proj(y)
        9. Return y
        """
        # TODO: Implement forward using F.scaled_dot_product_attention
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
