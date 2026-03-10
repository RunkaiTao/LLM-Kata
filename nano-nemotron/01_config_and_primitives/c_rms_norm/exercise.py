"""
Root Mean Square Layer Normalization (RMSNorm).

NemotronH uses RMSNorm (not LayerNorm) for all normalization layers.
RMSNorm is simpler and faster than LayerNorm because it skips the mean
subtraction step — it only normalizes by the root mean square:

    RMSNorm(x) = weight * x / sqrt(mean(x^2) + eps)

NemotronH uses "pre-norm" architecture where normalization comes BEFORE
each mixer layer (not after). The decoder layers use a fused add-then-norm
pattern for the residual stream:

    If no residual:  residual = x;  normed = RMSNorm(x)
    If residual:     residual = residual + x;  normed = RMSNorm(residual)

This fused pattern avoids materializing an intermediate tensor.

Reference: vllm/vllm/model_executor/layers/layernorm.py — RMSNorm
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# YOUR TASK: Implement RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Args:
            hidden_size: Dimension of the last axis to normalize over.
            eps: Small constant for numerical stability.

        Steps:
        1. Create self.weight as an nn.Parameter of ones with shape (hidden_size,)
           (use nn.Parameter and torch.ones)
        2. Store eps as self.variance_epsilon
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.weight = ...             (nn.Parameter of torch.ones(hidden_size))
        # Step 2: self.variance_epsilon = ...   (store eps)
        pass

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (..., hidden_size).
            residual: Optional residual tensor of same shape as x.

        Returns:
            If residual is None: normalized tensor of same shape.
            If residual is provided: (normalized tensor, new_residual) where
                new_residual = residual + x.

        Steps:
        1. If residual is provided:
           - Compute new_residual = residual + x (element-wise addition)
           - Set x = new_residual (normalize the sum)
        2. Else: set new_residual = None
        3. Convert x to float32 for numerical stability (use x.to(torch.float32))
        4. Compute the variance: mean of x^2 along the last dimension, keeping dims
           (use .pow(2).mean(-1, keepdim=True))
        5. Compute x_normed = x * rsqrt(variance + eps) (use torch.rsqrt)
        6. Multiply by self.weight and cast back to the original input dtype
        7. If residual was provided, return (normed_output, new_residual) as a tuple
           If residual was None, return just normed_output
        """
        # TODO: Implement forward following the steps above
        # Step 1: if residual is not None: new_residual = ...; x = new_residual
        # Step 2: else: new_residual = None
        # Step 3: x = ...           (convert to float32)
        # Step 4: variance = ...    (x.pow(2).mean(-1, keepdim=True))
        # Step 5: x = ...           (x * torch.rsqrt(variance + self.variance_epsilon))
        # Step 6: x = ...           (multiply by self.weight, cast back to input dtype)
        # Step 7: return (x, new_residual) if residual was provided, else x
        pass


# Run tests:
# pytest nano-nemotron/01_config_and_primitives/c_rms_norm/test_exercise.py -v
