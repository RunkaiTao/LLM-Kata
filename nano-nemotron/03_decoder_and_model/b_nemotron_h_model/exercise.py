"""
Full NemotronH Hybrid Model — assembling all layer types.

This exercise brings together all building blocks into the complete NemotronH
backbone. The key innovation is the HYBRID layer pattern — instead of using
the same layer type throughout (like GPT-2's uniform attention+MLP blocks),
NemotronH mixes four different layer types in a specific pattern:

    "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"

Where:  M = Mamba2,  - = MLP,  * = Attention,  E = MoE

The model architecture:
    1. Token embedding: vocab_size -> hidden_size
    2. 52 hybrid layers dispatched by pattern character
    3. Final RMSNorm
    4. Each layer follows the pre-norm + residual pattern

The forward pass maintains a (hidden_states, residual) pair that flows
through all layers, with the residual stream accumulating across layers.

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 560-639
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
RMSNorm = load("01_config_and_primitives", "c_rms_norm").RMSNorm
NemotronHMLP = load("02_layer_types", "a_mlp").NemotronHMLP
NemotronHAttention = load("02_layer_types", "b_gqa_attention").NemotronHAttention
Mamba2Mixer = load("02_layer_types", "c_mamba2").Mamba2Mixer
NemotronHMoE = load("02_layer_types", "d_moe").NemotronHMoE
NemotronHDecoderLayer = load("03_decoder_and_model", "a_decoder_layer").NemotronHDecoderLayer


# ---------------------------------------------------------------------------
# Layer type dispatch map
# ---------------------------------------------------------------------------
# Maps each pattern character to its mixer constructor.
# Each constructor takes (config,) and returns an nn.Module.
MIXER_TYPES = {
    "M": Mamba2Mixer,       # Mamba2 SSM
    "-": NemotronHMLP,      # Feed-forward MLP
    "*": NemotronHAttention, # Grouped Query Attention
    "E": NemotronHMoE,      # Mixture of Experts
}


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHModel
# ---------------------------------------------------------------------------

class NemotronHModel(nn.Module):
    """NemotronH backbone: embedding -> hybrid layers -> final norm."""

    def __init__(self, config: "NemotronHConfig"):
        """
        Args:
            config: NemotronHConfig instance.

        Steps:
        1. Store config as self.config
        2. Create self.embed_tokens as nn.Embedding(config.vocab_size, config.hidden_size)
        3. Build self.layers as nn.ModuleList by iterating over config.hybrid_override_pattern:
           For each character in the pattern:
           a. Look up the mixer class in MIXER_TYPES
           b. Instantiate the mixer: mixer = MixerClass(config)
           c. Wrap it in a NemotronHDecoderLayer: NemotronHDecoderLayer(config, mixer)
           (the list should have config.num_hidden_layers elements)
        4. Create self.norm_f as RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.config = ...
        # Step 2: self.embed_tokens = ...  (nn.Embedding: vocab_size -> hidden_size)
        # Step 3: self.layers = nn.ModuleList(...)  (iterate pattern, wrap each mixer in NemotronHDecoderLayer)
        # Step 4: self.norm_f = ...  (RMSNorm with config.hidden_size, eps=config.layer_norm_epsilon)
        pass

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices of shape (B, T).

        Returns:
            Hidden states of shape (B, T, hidden_size).

        Steps:
        1. Embed input tokens: hidden_states = self.embed_tokens(input_ids)
           -> (B, T, hidden_size)
        2. Initialize residual = None
        3. Loop through each layer in self.layers:
           hidden_states, residual = layer(hidden_states, residual)
        4. Apply final normalization with the residual:
           hidden_states, _ = self.norm_f(hidden_states, residual)
        5. Return hidden_states
        """
        # TODO: Implement forward following the steps above
        # Step 1: hidden_states = ...  (embed input_ids via self.embed_tokens)
        # Step 2: residual = None
        # Step 3: for layer in self.layers: hidden_states, residual = layer(hidden_states, residual)
        # Step 4: hidden_states, _ = ...  (apply self.norm_f with residual)
        # return hidden_states
        pass


# Run tests:
# pytest nano-nemotron/03_decoder_and_model/b_nemotron_h_model/test_exercise.py -v
