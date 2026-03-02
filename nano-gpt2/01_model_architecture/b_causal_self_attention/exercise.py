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
        1. Assert config.n_embd % config.n_head == 0
        2. Create self.c_attn as nn.Linear(config.n_embd, 3 * config.n_embd)
           — this single layer computes Q, K, V for ALL heads in one shot
        3. Create self.c_proj as nn.Linear(config.n_embd, config.n_embd)
           — output projection after concatenating heads
        4. Set self.c_proj.NANOGPT_SCALE_INIT = 1
           — flag used later by GPT._init_weights for scaled initialization
        5. Store self.n_head = config.n_head and self.n_embd = config.n_embd
        6. Register a buffer named 'bias' containing a lower-triangular causal mask
           of shape (1, 1, config.block_size, config.block_size)
           — use torch.tril(torch.ones(...))
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C) where C = n_embd.

        Returns:
            Output tensor of shape (B, T, C).

        Steps:
        1. Extract B, T, C from x.size()
        2. Compute qkv = self.c_attn(x) -> (B, T, 3*C)
        3. Split qkv into q, k, v using qkv.split(self.n_embd, dim=2) -> each (B, T, C)
        4. Reshape k: k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) -> (B, nh, T, hs)
        5. Reshape q the same way -> (B, nh, T, hs)
        6. Reshape v the same way -> (B, nh, T, hs)
        7. Compute attention scores: att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
           -> (B, nh, T, T)
        8. Apply causal mask: att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        9. Normalize: att = F.softmax(att, dim=-1)
        10. Apply attention to values: y = att @ v -> (B, nh, T, hs)
        11. Reassemble heads: y = y.transpose(1, 2).contiguous().view(B, T, C) -> (B, T, C)
        12. Output projection: y = self.c_proj(y)
        13. Return y
        """
        # TODO: Implement forward following the steps above
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
