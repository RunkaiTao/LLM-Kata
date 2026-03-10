import math

import torch
import pytest
from torch.nn import functional as F
from exercise import reference_attention

BATCH = 2
HEADS = 4
SEQ_LEN = 8
HEAD_DIM = 16


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    Q = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    K = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    V = torch.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    return Q, K, V


class TestReferenceAttention:
    def test_output_shape(self, qkv):
        """Output must have the same shape as Q: (B, H, T, D)."""
        Q, K, V = qkv
        out = reference_attention(Q, K, V)
        assert out.shape == (BATCH, HEADS, SEQ_LEN, HEAD_DIM), \
            f"Expected {(BATCH, HEADS, SEQ_LEN, HEAD_DIM)}, got {out.shape}"

    def test_causal_masking(self, qkv):
        """Position i must not be influenced by tokens at positions j > i."""
        Q, K, V = qkv
        Q2, K2, V2 = Q.clone(), K.clone(), V.clone()
        # Corrupt the last 4 positions
        Q2[:, :, 4:, :] = torch.randn_like(Q2[:, :, 4:, :])
        K2[:, :, 4:, :] = torch.randn_like(K2[:, :, 4:, :])
        V2[:, :, 4:, :] = torch.randn_like(V2[:, :, 4:, :])

        out1 = reference_attention(Q, K, V, causal=True)
        out2 = reference_attention(Q2, K2, V2, causal=True)

        # Positions 0-3 should be unaffected by changes at positions 4-7
        assert torch.allclose(out1[:, :, :4, :], out2[:, :, :4, :], atol=1e-5), \
            "Causal masking failed: early positions are affected by future tokens"

    def test_attention_weights_sum_to_one(self, qkv):
        """Attention weights (softmax output) must sum to 1 over the key dimension."""
        Q, K, V = qkv
        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = Q @ K.transpose(-2, -1) * scale
        weights = F.softmax(scores, dim=-1)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_non_causal_vs_causal_differ(self, qkv):
        """Non-causal and causal outputs must differ (for T > 1)."""
        Q, K, V = qkv
        out_nc = reference_attention(Q, K, V, causal=False)
        out_c = reference_attention(Q, K, V, causal=True)
        assert not torch.allclose(out_nc, out_c), \
            "Causal and non-causal outputs should differ"

    def test_matches_pytorch_sdpa(self, qkv):
        """Must match torch.nn.functional.scaled_dot_product_attention."""
        Q, K, V = qkv
        expected_nc = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        expected_c = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out_nc = reference_attention(Q, K, V, causal=False)
        out_c = reference_attention(Q, K, V, causal=True)
        assert torch.allclose(out_nc, expected_nc, atol=1e-5), \
            "Non-causal output doesn't match F.scaled_dot_product_attention"
        assert torch.allclose(out_c, expected_c, atol=1e-5), \
            "Causal output doesn't match F.scaled_dot_product_attention"
