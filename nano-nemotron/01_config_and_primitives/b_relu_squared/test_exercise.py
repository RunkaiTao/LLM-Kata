import torch
import pytest
from exercise import ReLUSquaredActivation


@pytest.fixture
def act():
    return ReLUSquaredActivation()


class TestReLUSquaredActivation:
    def test_is_nn_module(self, act):
        assert isinstance(act, torch.nn.Module)

    def test_positive_input(self, act):
        """relu(2.0)^2 = 4.0"""
        x = torch.tensor([2.0])
        result = act(x)
        assert torch.allclose(result, torch.tensor([4.0]))

    def test_negative_input_is_zero(self, act):
        """relu(-1.0)^2 = 0.0"""
        x = torch.tensor([-1.0, -5.0, -0.1])
        result = act(x)
        assert torch.allclose(result, torch.zeros(3))

    def test_zero_input(self, act):
        x = torch.tensor([0.0])
        result = act(x)
        assert torch.allclose(result, torch.tensor([0.0]))

    def test_output_shape_preserved(self, act):
        x = torch.randn(2, 16, 64)
        result = act(x)
        assert result.shape == (2, 16, 64)

    def test_output_non_negative(self, act):
        """ReLU^2 output should always be >= 0."""
        x = torch.randn(100)
        result = act(x)
        assert (result >= 0).all()

    def test_squared_not_just_relu(self, act):
        """Should be relu^2, not just relu. relu(3) = 3, relu^2(3) = 9."""
        x = torch.tensor([3.0])
        result = act(x)
        assert torch.allclose(result, torch.tensor([9.0]))

    def test_gradient_flows(self, act):
        x = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
        result = act(x)
        result.sum().backward()
        assert x.grad is not None
        # d/dx relu(x)^2 = 2*relu(x) for x>0, 0 for x<0
        # x=1: grad=2*1=2, x=-1: grad=0, x=2: grad=2*2=4
        expected = torch.tensor([2.0, 0.0, 4.0])
        assert torch.allclose(x.grad, expected)

    def test_mixed_values(self, act):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = act(x)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 4.0])
        assert torch.allclose(result, expected)
