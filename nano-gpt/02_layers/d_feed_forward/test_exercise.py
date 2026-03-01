import torch
import pytest
from exercise import FeedForward

N_EMBD = 32
BATCH_SIZE = 2
SEQ_LEN = 8


class TestFeedForward:
    @pytest.fixture
    def ffn(self):
        torch.manual_seed(42)
        return FeedForward(N_EMBD, dropout=0.0)

    def test_output_shape(self, ffn):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, N_EMBD)
        out = ffn(x)
        assert out.shape == (BATCH_SIZE, SEQ_LEN, N_EMBD)

    def test_expansion_factor(self, ffn):
        """First linear should expand 4x, second should compress back"""
        layers = list(ffn.net.children())
        linear_layers = [l for l in layers if isinstance(l, torch.nn.Linear)]
        assert len(linear_layers) == 2
        assert linear_layers[0].in_features == N_EMBD
        assert linear_layers[0].out_features == 4 * N_EMBD
        assert linear_layers[1].in_features == 4 * N_EMBD
        assert linear_layers[1].out_features == N_EMBD

    def test_parameter_count(self, ffn):
        """Check total parameter count"""
        param_count = sum(p.numel() for p in ffn.parameters())
        # Linear1: n_embd * 4*n_embd + 4*n_embd = 32*128 + 128 = 4224
        # Linear2: 4*n_embd * n_embd + n_embd = 128*32 + 32 = 4128
        expected = (N_EMBD * 4 * N_EMBD + 4 * N_EMBD) + (4 * N_EMBD * N_EMBD + N_EMBD)
        assert param_count == expected

    def test_deterministic(self, ffn):
        x = torch.randn(1, 4, N_EMBD)
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.equal(out1, out2)
