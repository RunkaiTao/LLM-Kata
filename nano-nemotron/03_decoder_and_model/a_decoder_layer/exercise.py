"""
NemotronH Decoder Layer — Pre-Norm Residual Wrapper.

Every layer in NemotronH (MLP, Attention, Mamba2, MoE) follows the same
pre-normalization residual pattern:

    If first layer (residual is None):
        residual = hidden_states
        hidden_states = RMSNorm(hidden_states)
    Else:
        hidden_states, residual = RMSNorm(hidden_states, residual)
        (i.e., residual = residual + hidden_states; hidden_states = norm(residual))

    hidden_states = mixer(hidden_states)
    return hidden_states, residual

The residual stream is maintained SEPARATELY and accumulated across layers.
The mixer (MLP/Attention/Mamba2/MoE) only sees the normalized hidden states.
This "pre-norm" design is more stable than post-norm for deep networks.

There are 4 decoder layer types, but they all follow this same pattern.
The only difference is which mixer module they wrap. We implement a generic
decoder layer that accepts any mixer.

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 282-549
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
RMSNorm = load("01_config_and_primitives", "c_rms_norm").RMSNorm


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHDecoderLayer
# ---------------------------------------------------------------------------

class NemotronHDecoderLayer(nn.Module):
    """Generic decoder layer: pre-norm + mixer + residual."""

    def __init__(self, config: "NemotronHConfig", mixer: nn.Module):
        """
        Args:
            config: NemotronHConfig instance.
            mixer: Pre-constructed layer module (MLP, Attention, Mamba2, or MoE).

        Steps:
        1. Store the mixer module as self.mixer
        2. Create self.norm as RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.mixer = ...  (store the pre-constructed mixer module)
        # Step 2: self.norm = ...   (RMSNorm with config.hidden_size, eps=config.layer_norm_epsilon)
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input of shape (B, T, hidden_size).
            residual: Residual stream tensor, or None for first layer.

        Returns:
            Tuple of (hidden_states, residual), both of shape (B, T, hidden_size).

        Steps:
        1. Apply pre-norm with residual handling:
           - If residual is None (first layer):
               residual = hidden_states
               hidden_states = self.norm(hidden_states)
           - If residual is not None (subsequent layers):
               hidden_states, residual = self.norm(hidden_states, residual)
               (this does: residual_new = residual + hidden_states,
                hidden_states = RMSNorm(residual_new))
        2. Pass normalized hidden_states through self.mixer
        3. Return (hidden_states, residual) as a tuple
        """
        # TODO: Implement forward following the steps above
        # Step 1: if residual is None: residual = ...; hidden_states = self.norm(...)
        #         else: hidden_states, residual = self.norm(hidden_states, residual)
        # Step 2: hidden_states = ...  (pass through self.mixer)
        # return (hidden_states, residual)
        pass


# Run tests:
# pytest nano-nemotron/03_decoder_and_model/a_decoder_layer/test_exercise.py -v
