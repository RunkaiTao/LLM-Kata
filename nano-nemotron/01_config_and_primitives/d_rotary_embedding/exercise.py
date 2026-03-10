"""
Rotary Position Embedding (RoPE) for NemotronH.

RoPE encodes position information by rotating query and key vectors in the
attention mechanism. Unlike absolute position embeddings (added to token
embeddings), RoPE applies a position-dependent rotation in the attention
computation itself.

Key insight: the dot product of two rotated vectors depends only on their
*relative* position, not absolute positions. This gives the model a natural
notion of distance between tokens.

How it works:
    1. Compute inverse frequencies: theta_i = 1 / (base^(2i/dim))
       where i = 0, 1, ..., dim/2-1 and base = rope_theta (10000.0)
    2. For each position pos, compute angles: pos * theta_i
    3. Build cos/sin tables from these angles
    4. Rotate Q and K: q_rot = q * cos + rotate_half(q) * sin

The rotation pairs adjacent dimensions: (x0, x1), (x2, x3), ...
rotate_half swaps each pair and negates the first element:
    [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]

Reference: vllm/vllm/model_executor/layers/rotary_embedding.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement rotate_half, RotaryEmbedding, and apply_rotary_pos_emb
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate pairs of dimensions by swapping halves and negating.

    Split the last dimension into two halves (x1, x2), then return [-x2, x1].
    This is the rotation operation used in RoPE.

    Args:
        x: Tensor of shape (..., dim) where dim is even.

    Returns:
        Rotated tensor of same shape.

    Steps:
    1. Split x along last dim into two halves: x1 = x[..., :half], x2 = x[..., half:]
    2. Return torch.cat([-x2, x1], dim=-1)
    """
    # TODO: Implement rotate_half following the steps above
    # Step 1: x1 = ..., x2 = ...  (split x along last dim into two halves)
    # Step 2: return ...           (torch.cat([-x2, x1], dim=-1))
    pass


class RotaryEmbedding(nn.Module):
    """Computes cos/sin position embeddings for RoPE."""

    def __init__(self, config: "NemotronHConfig"):
        """
        Args:
            config: NemotronHConfig instance with head_dim, max_position_embeddings, rope_theta.

        Steps:
        1. Store self.dim = config.head_dim
        2. Store self.max_position_embeddings = config.max_position_embeddings
        3. Store self.base = config.rope_theta
        4. Compute inverse frequencies:
           inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
           This gives dim/2 frequencies that decrease geometrically.
        5. Register inv_freq as a buffer (not a parameter):
           self.register_buffer("inv_freq", inv_freq)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.dim = ...                       (config.head_dim)
        # Step 2: self.max_position_embeddings = ...   (config.max_position_embeddings)
        # Step 3: self.base = ...                      (config.rope_theta)
        # Step 4: inv_freq = ...                       (1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        # Step 5: self.register_buffer("inv_freq", inv_freq)
        pass

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, num_heads, T, head_dim) — for dtype/device only.
            position_ids: Position indices (B, T) or (1, T).

        Returns:
            Tuple of (cos, sin), each of shape (B, T, head_dim).

        Steps:
        1. Expand inv_freq for batched matmul:
           inv_freq_expanded = self.inv_freq.unsqueeze(0).unsqueeze(0)  -> (1, 1, dim/2)
        2. Expand position_ids for matmul:
           position_ids_expanded = position_ids.unsqueeze(-1).float()   -> (B, T, 1)
        3. Compute angles:
           freqs = position_ids_expanded @ inv_freq_expanded             -> (B, T, dim/2)
        4. Build full rotation:
           emb = torch.cat([freqs, freqs], dim=-1)                      -> (B, T, dim)
        5. Return (cos(emb), sin(emb)) as a tuple, cast to x.dtype
        """
        # TODO: Implement forward following the steps above
        # Step 1: inv_freq_expanded = ...       (unsqueeze inv_freq to (1, 1, dim/2))
        # Step 2: position_ids_expanded = ...   (unsqueeze position_ids to (B, T, 1), float)
        # Step 3: freqs = ...                   (position_ids_expanded @ inv_freq_expanded)
        # Step 4: emb = ...                     (torch.cat([freqs, freqs], dim=-1))
        # Step 5: return (cos(emb).to(x.dtype), sin(emb).to(x.dtype))
        pass


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to query and key tensors.

    Args:
        q: Query tensor of shape (B, num_heads, T, head_dim).
        k: Key tensor of shape (B, num_kv_heads, T, head_dim).
        cos: Cosine table of shape (B, T, head_dim).
        sin: Sine table of shape (B, T, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs.

    Steps:
    1. Reshape cos and sin for broadcasting with (B, num_heads, T, head_dim):
       cos = cos.unsqueeze(1)  -> (B, 1, T, head_dim)
       sin = sin.unsqueeze(1)  -> (B, 1, T, head_dim)
    2. Rotate q: q_embed = q * cos + rotate_half(q) * sin
    3. Rotate k: k_embed = k * cos + rotate_half(k) * sin
    4. Return (q_embed, k_embed)
    """
    # TODO: Implement apply_rotary_pos_emb following the steps above
    # Step 1: cos = ..., sin = ...    (unsqueeze both to (B, 1, T, head_dim))
    # Step 2: q_embed = ...           (q * cos + rotate_half(q) * sin)
    # Step 3: k_embed = ...           (k * cos + rotate_half(k) * sin)
    # return (q_embed, k_embed)
    pass


# Run tests:
# pytest nano-nemotron/01_config_and_primitives/d_rotary_embedding/test_exercise.py -v
