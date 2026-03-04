"""
ReLU Squared (ReLU^2) activation function.

NemotronH uses ReLU squared instead of GELU or SiLU for its MLP layers.
This is a simple but effective activation: apply ReLU, then square the result.

    ReLU^2(x) = relu(x)^2

This produces smoother gradients than plain ReLU while maintaining sparsity
(negative inputs still produce zero). It was found to work well in large
language models, particularly in the Nemotron family.

Reference: vllm/vllm/model_executor/layers/activation.py — ReLUSquaredActivation
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# YOUR TASK: Implement ReLUSquaredActivation as an nn.Module
# ---------------------------------------------------------------------------

class ReLUSquaredActivation(nn.Module):
    """Activation function: relu(x) ** 2"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape.

        Returns:
            relu(x) ** 2, same shape as input.

        Steps:
        1. Apply torch.relu to x
        2. Square the result (use torch.square or ** operator)
        3. Return the squared result
        """
        # TODO: Implement forward following the steps above
        # Step 1: x = ...  (apply torch.relu)
        # Step 2: x = ...  (square with ** 2 or torch.square)
        # return x
        pass


# Run tests:
# pytest nano-nemotron/01_config_and_primitives/b_relu_squared/test_exercise.py -v
