"""Tests for a3_online_softmax_block."""
import pytest
import torch

triton = pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from exercise import online_softmax_block


class TestOnlineSoftmaxBlock:
    def test_output_shapes(self):
        QK = torch.randn(16, 32, dtype=torch.float32, device="cuda")
        m_i = torch.full((16,), float("-inf"), device="cuda")
        l_i = torch.zeros(16, device="cuda")
        m_out, l_out, P, alpha = online_softmax_block(QK, m_i, l_i)
        assert m_out.shape == (16,)
        assert l_out.shape == (16,)
        assert P.shape == (16, 32)
        assert alpha.shape == (16,)

    def test_first_block_m_equals_row_max(self):
        """When m_i=-inf, m_new should be the row maxima."""
        QK = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device="cuda")
        m_i = torch.full((2,), float("-inf"), device="cuda")
        l_i = torch.zeros(2, device="cuda")
        m_out, _, _, _ = online_softmax_block(QK, m_i, l_i)
        assert torch.allclose(m_out, torch.tensor([2.0, 3.0], device="cuda"), atol=1e-5)

    def test_first_block_alpha_zero(self):
        """When m_i=-inf, alpha should be 0."""
        QK = torch.randn(16, 16, dtype=torch.float32, device="cuda")
        m_i = torch.full((16,), float("-inf"), device="cuda")
        l_i = torch.zeros(16, device="cuda")
        _, _, _, alpha = online_softmax_block(QK, m_i, l_i)
        assert torch.allclose(alpha, torch.zeros(16, device="cuda"), atol=1e-6)

    def test_p_block_is_exp_shifted(self):
        """P_block should equal exp(QK - m_new)."""
        QK = torch.tensor([[3.0, 1.0]], device="cuda")
        m_i = torch.full((1,), float("-inf"), device="cuda")
        l_i = torch.zeros(1, device="cuda")
        m_out, _, P, _ = online_softmax_block(QK, m_i, l_i)
        # m_out = 3.0, P = [exp(0), exp(-2)]
        expected = torch.tensor([[1.0, torch.exp(torch.tensor(-2.0)).item()]], device="cuda")
        assert torch.allclose(P, expected, atol=1e-5)

    def test_l_new_first_block(self):
        """On first block, l_new = sum(P_block, dim=1)."""
        QK = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device="cuda")
        m_i = torch.full((2,), float("-inf"), device="cuda")
        l_i = torch.zeros(2, device="cuda")
        _, l_out, P, _ = online_softmax_block(QK, m_i, l_i)
        expected_l = P.sum(dim=1)
        assert torch.allclose(l_out, expected_l, atol=1e-5)

    def test_two_blocks_yield_softmax(self):
        """Processing two halves should produce correct softmax weights."""
        torch.manual_seed(42)
        x = torch.randn(16, 32, device="cuda")
        half = 16

        # Block 1
        m_i = torch.full((16,), float("-inf"), device="cuda")
        l_i = torch.zeros(16, device="cuda")
        m_i, l_i, P1, _ = online_softmax_block(x[:, :half], m_i, l_i)

        # Block 2
        m_i, l_i, P2, alpha = online_softmax_block(x[:, half:], m_i, l_i)

        # Reconstruct full softmax
        P1_rescaled = P1 * alpha[:, None]
        P_full = torch.cat([P1_rescaled, P2], dim=1) / l_i[:, None]
        expected = torch.softmax(x, dim=1)
        assert torch.allclose(P_full, expected, atol=1e-4)

    def test_alpha_leq_one(self):
        """alpha = exp(m_old - m_new) should always be <= 1."""
        torch.manual_seed(99)
        m_i = torch.tensor([1.0, 2.0, 3.0, 0.5] * 4, device="cuda")
        l_i = torch.ones(16, device="cuda")
        QK = torch.randn(16, 16, device="cuda") + 5
        _, _, _, alpha = online_softmax_block(QK, m_i, l_i)
        assert (alpha <= 1.0 + 1e-5).all()

    def test_m_never_decreases(self):
        """Running max should be monotonically non-decreasing across blocks."""
        torch.manual_seed(7)
        m_i = torch.full((16,), float("-inf"), device="cuda")
        l_i = torch.zeros(16, device="cuda")
        for _ in range(4):
            QK = torch.randn(16, 16, device="cuda")
            prev_m = m_i.clone()
            m_i, l_i, _, _ = online_softmax_block(QK, m_i, l_i)
            assert (m_i >= prev_m).all()
