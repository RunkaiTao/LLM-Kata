"""
NemotronH MLP (Feed-Forward) Layer.

The simplest layer type in NemotronH, marked as "-" in the hybrid pattern.
Unlike GPT-2's gated MLP (which uses GELU and 4x expansion), NemotronH's MLP
uses ReLU squared activation and a configurable intermediate_size (1856 by default).

There is NO gating here (unlike LLaMA's SwiGLU which has a gate projection).
The architecture is a simple two-layer feed-forward:

    x -> up_proj -> ReLU^2 -> down_proj -> output

    up_proj:   hidden_size (2688)  -> intermediate_size (1856)
    down_proj: intermediate_size (1856) -> hidden_size (2688)

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 87-124
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
ReLUSquaredActivation = load("01_config_and_primitives", "b_relu_squared").ReLUSquaredActivation


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHMLP
# ---------------------------------------------------------------------------

class NemotronHMLP(nn.Module):
    """Simple feed-forward MLP with ReLU squared activation."""

    def __init__(
        self,
        config: "NemotronHConfig",
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
    ):
        """
        Args:
            config: NemotronHConfig instance.
            hidden_size: Override hidden dimension (defaults to config.hidden_size).
            intermediate_size: Override intermediate dimension (defaults to config.intermediate_size).

        Steps:
        1. If hidden_size is None, use config.hidden_size
        2. If intermediate_size is None, use config.intermediate_size
        3. Create self.up_proj as nn.Linear(hidden_size, intermediate_size, bias=False)
        4. Create self.down_proj as nn.Linear(intermediate_size, hidden_size, bias=False)
        5. Create self.act_fn as a ReLUSquaredActivation instance
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: hidden_size = ...        (default to config.hidden_size if None)
        # Step 2: intermediate_size = ...  (default to config.intermediate_size if None)
        # Step 3: self.up_proj = ...       (nn.Linear: hidden_size -> intermediate_size, no bias)
        # Step 4: self.down_proj = ...     (nn.Linear: intermediate_size -> hidden_size, no bias)
        # Step 5: self.act_fn = ...        (ReLUSquaredActivation instance)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).

        Steps:
        1. Pass x through self.up_proj  -> shape becomes (..., intermediate_size)
        2. Apply self.act_fn            -> same shape
        3. Pass through self.down_proj  -> shape becomes (..., hidden_size)
        4. Return the result
        """
        # TODO: Implement forward following the steps above
        # Step 1: x = ...  (pass through up_proj)
        # Step 2: x = ...  (apply act_fn)
        # Step 3: x = ...  (pass through down_proj)
        # return x
        pass


# Run tests:
# pytest nano-nemotron/02_layer_types/a_mlp/test_exercise.py -v
