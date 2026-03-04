"""
Grouped Query Attention (GQA) with Rotary Position Embeddings for NemotronH.

NemotronH uses GQA where the number of Key/Value heads is less than Query heads.
This saves memory and compute compared to standard Multi-Head Attention (MHA):

    MHA:  num_kv_heads == num_attention_heads  (every Q head has its own K,V)
    GQA:  num_kv_heads <  num_attention_heads  (multiple Q heads share K,V groups)
    MQA:  num_kv_heads == 1                    (all Q heads share a single K,V)

Default NemotronH: 32 Q heads, 2 KV heads -> each KV group serves 16 Q heads.

The attention uses a single QKV projection (no bias), RoPE, and causal masking:

    qkv = qkv_proj(hidden_states)         # hidden -> Q + K + V
    q, k, v = split(qkv)                  # separate Q, K, V
    q, k = apply_rotary_pos_emb(q, k)     # position encoding via rotation
    attn_output = scaled_dot_product(q, k, v, causal=True)
    output = o_proj(attn_output)           # project back to hidden_size

For GQA, K and V must be expanded (repeated) to match the number of Q heads
before computing attention scores.

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 430-504
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
_rope_mod = load("01_config_and_primitives", "d_rotary_embedding")
RotaryEmbedding = _rope_mod.RotaryEmbedding
apply_rotary_pos_emb = _rope_mod.apply_rotary_pos_emb


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHAttention
# ---------------------------------------------------------------------------

class NemotronHAttention(nn.Module):
    """Grouped Query Attention with combined QKV projection and RoPE."""

    def __init__(self, config: "NemotronHConfig", layer_idx: int = 0):
        """
        Args:
            config: NemotronHConfig instance.
            layer_idx: Layer index (unused here, kept for interface compatibility).

        Steps:
        1. Store from config:
           - self.hidden_size = config.hidden_size
           - self.num_heads = config.num_attention_heads       (total Q heads)
           - self.num_kv_heads = config.num_key_value_heads    (total KV heads)
           - self.head_dim = config.head_dim                   (dimension per head)
        2. Compute derived sizes:
           - self.q_size = self.num_heads * self.head_dim      (total Q dimension)
           - self.kv_size = self.num_kv_heads * self.head_dim  (total KV dimension)
           - self.scaling = self.head_dim ** -0.5              (1/sqrt(head_dim))
           - self.num_kv_groups = self.num_heads // self.num_kv_heads
        3. Create self.qkv_proj as nn.Linear(hidden_size, q_size + 2 * kv_size, bias=False)
        4. Create self.o_proj as nn.Linear(q_size, hidden_size, bias=False)
        5. Create self.rotary_emb as RotaryEmbedding(config)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.hidden_size = ..., self.num_heads = ..., self.num_kv_heads = ..., self.head_dim = ...
        # Step 2: self.q_size = ..., self.kv_size = ..., self.scaling = ..., self.num_kv_groups = ...
        # Step 3: self.qkv_proj = ...   (nn.Linear: hidden_size -> q_size + 2*kv_size, no bias)
        # Step 4: self.o_proj = ...     (nn.Linear: q_size -> hidden_size, no bias)
        # Step 5: self.rotary_emb = ... (RotaryEmbedding(config))
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input of shape (B, T, hidden_size).

        Returns:
            Output of shape (B, T, hidden_size).

        Steps:
        1. Compute qkv = self.qkv_proj(hidden_states)  -> (..., q_size + 2*kv_size)
        2. Split qkv into q, k, v along last dim using sizes [q_size, kv_size, kv_size]
           (use torch.split or Tensor.split)
        3. Get B (batch) and T (seq_len) from hidden_states.shape
        4. Reshape q to (B, T, num_heads, head_dim) then transpose to (B, num_heads, T, head_dim)
        5. Reshape k to (B, T, num_kv_heads, head_dim) then transpose to (B, num_kv_heads, T, head_dim)
        6. Reshape v to (B, T, num_kv_heads, head_dim) then transpose to (B, num_kv_heads, T, head_dim)
        7. Apply RoPE to Q and K (before GQA expansion):
           a. Compute position_ids = torch.arange(T, device=q.device).unsqueeze(0)  -> (1, T)
           b. cos, sin = self.rotary_emb(q, position_ids)
           c. q, k = apply_rotary_pos_emb(q, k, cos, sin)
        8. Expand K and V for GQA: repeat each KV head for its group of Q heads
           k = k.repeat_interleave(self.num_kv_groups, dim=1)  -> (B, num_heads, T, head_dim)
           v = v.repeat_interleave(self.num_kv_groups, dim=1)  -> (B, num_heads, T, head_dim)
        9. Compute attention scores: att = (q @ k.transpose(-2, -1)) * self.scaling
        10. Apply causal mask: create a lower-triangular boolean mask of shape (T, T),
            fill positions where mask is False with -inf
            (use torch.tril, torch.ones, .masked_fill)
        11. Apply softmax over last dimension (use F.softmax)
        12. Compute attention output: y = att @ v  -> (B, num_heads, T, head_dim)
        13. Transpose back: y = y.transpose(1, 2).contiguous().view(B, T, q_size)
        14. Project through o_proj and return
        """
        # TODO: Implement forward following the steps above
        # Step 1:  qkv = ...            (project hidden_states through qkv_proj)
        # Step 2:  q, k, v = ...        (split qkv along last dim: [q_size, kv_size, kv_size])
        # Step 3:  B, T = ...           (from hidden_states.shape)
        # Step 4:  q = ...              (reshape to (B, T, num_heads, head_dim), transpose to (B, num_heads, T, head_dim))
        # Step 5:  k = ...              (reshape to (B, T, num_kv_heads, head_dim), transpose similarly)
        # Step 6:  v = ...              (reshape to (B, T, num_kv_heads, head_dim), transpose similarly)
        # Step 7:  position_ids = ...   (torch.arange(T)); cos, sin = ...; q, k = apply_rotary_pos_emb(...)
        # Step 8:  k = ..., v = ...     (repeat_interleave for GQA expansion)
        # Step 9:  att = ...            (q @ k.transpose(-2, -1) * self.scaling)
        # Step 10: att = ...            (apply causal mask with .masked_fill, -inf)
        # Step 11: att = ...            (F.softmax over last dim)
        # Step 12: y = ...              (att @ v)
        # Step 13: y = ...              (transpose, contiguous, view back to (B, T, q_size))
        # Step 14: y = ...              (project through o_proj)
        # return y
        pass


# Run tests:
# pytest nano-nemotron/02_layer_types/b_gqa_attention/test_exercise.py -v
