import torch
import pytest
from exercise import RMSNorm

HIDDEN_SIZE = 64
EPS = 1e-5


@pytest.fixture
def norm():
    return RMSNorm(HIDDEN_SIZE, eps=EPS)


class TestRMSNormInit:
    def test_is_nn_module(self, norm):
        assert isinstance(norm, torch.nn.Module)

    def test_weight_shape(self, norm):
        assert norm.weight.shape == (HIDDEN_SIZE,)

    def test_weight_initialized_to_ones(self, norm):
        assert torch.allclose(norm.weight, torch.ones(HIDDEN_SIZE))

    def test_weight_is_parameter(self, norm):
        assert isinstance(norm.weight, torch.nn.Parameter)

    def test_eps_stored(self, norm):
        assert norm.variance_epsilon == EPS


class TestRMSNormForwardBasic:
    def test_output_shape_2d(self, norm):
        x = torch.randn(4, HIDDEN_SIZE)
        out = norm(x)
        assert out.shape == (4, HIDDEN_SIZE)

    def test_output_shape_3d(self, norm):
        x = torch.randn(2, 8, HIDDEN_SIZE)
        out = norm(x)
        assert out.shape == (2, 8, HIDDEN_SIZE)

    def test_output_dtype_preserved(self, norm):
        x = torch.randn(2, HIDDEN_SIZE)
        out = norm(x)
        assert out.dtype == x.dtype

    def test_unit_rms_after_norm(self, norm):
        """After RMSNorm (with weight=1), the RMS of each vector should be ~1."""
        x = torch.randn(100, HIDDEN_SIZE)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones(100), atol=1e-4)

    def test_deterministic(self, norm):
        x = torch.randn(2, HIDDEN_SIZE)
        out1 = norm(x)
        out2 = norm(x)
        assert torch.equal(out1, out2)


class TestRMSNormResidual:
    def test_returns_tuple_when_residual_provided(self, norm):
        x = torch.randn(2, HIDDEN_SIZE)
        residual = torch.randn(2, HIDDEN_SIZE)
        result = norm(x, residual)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_tensor_when_no_residual(self, norm):
        x = torch.randn(2, HIDDEN_SIZE)
        result = norm(x)
        assert isinstance(result, torch.Tensor)

    def test_residual_is_sum(self, norm):
        """new_residual should be residual + x."""
        x = torch.randn(2, HIDDEN_SIZE)
        residual = torch.randn(2, HIDDEN_SIZE)
        normed, new_residual = norm(x, residual)
        expected_residual = residual + x
        assert torch.allclose(new_residual, expected_residual, atol=1e-6)

    def test_normed_output_from_sum(self, norm):
        """The normed output should be RMSNorm(residual + x)."""
        x = torch.randn(2, HIDDEN_SIZE)
        residual = torch.randn(2, HIDDEN_SIZE)
        normed, new_residual = norm(x, residual)
        # Manually compute expected
        combined = residual + x
        expected = norm(combined)
        assert torch.allclose(normed, expected, atol=1e-5)


class TestRMSNormGradients:
    def test_gradients_flow(self, norm):
        x = torch.randn(2, HIDDEN_SIZE, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_weight_gradients(self, norm):
        x = torch.randn(2, HIDDEN_SIZE)
        out = norm(x)
        out.sum().backward()
        assert norm.weight.grad is not None
