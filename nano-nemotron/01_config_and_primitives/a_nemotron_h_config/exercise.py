"""
NemotronH model configuration using a Python dataclass.

Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8

NemotronH is a hybrid architecture from NVIDIA that combines four layer types:
  M = Mamba2 (state space model for linear-time sequence processing)
  * = Attention (grouped query attention)
  - = MLP (simple feed-forward with ReLU squared)
  E = MoE (sparse mixture of experts)

These layers are arranged in a repeating pattern defined by hybrid_override_pattern.
The default pattern is "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
which gives 52 total layers: Mamba, MoE, and periodic attention layers (no plain MLP).

Reference: vllm/vllm/transformers_utils/configs/nemotron_h.py
"""
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# YOUR TASK: Define NemotronHConfig as a dataclass with actual model defaults
# ---------------------------------------------------------------------------

# TODO: Decorate the class with @dataclass
# TODO: Define the following fields with their default values:
#
#   General:
#     vocab_size: int = 131072        — 131k token vocabulary
#     hidden_size: int = 2688         — main hidden dimension
#     intermediate_size: int = 1856   — MLP intermediate dimension
#     num_hidden_layers: int = 52     — total number of layers
#     layer_norm_epsilon: float = 1e-5
#     tie_word_embeddings: bool = False
#
#   Hybrid layer pattern:
#     hybrid_override_pattern: str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
#       (each character maps to a layer type: M=Mamba2, *=Attention, -=MLP, E=MoE)
#
#   Attention:
#     num_attention_heads: int = 32   — total Q heads
#     head_dim: int = 128             — dimension per head
#     num_key_value_heads: int = 2    — KV heads (GQA: fewer KV than Q heads)
#     attention_bias: bool = False
#     max_position_embeddings: int = 262144  — 256K context window
#     rope_theta: float = 10000.0     — RoPE base frequency
#
#   MLP:
#     mlp_bias: bool = False
#
#   Mamba2 (State Space Model):
#     mamba_num_heads: int = 64       — SSM heads
#     mamba_head_dim: int = 64        — SSM head dimension
#     mamba_n_groups: int = 8         — number of SSM groups
#     ssm_state_size: int = 128       — state dimension per head
#     mamba_d_conv: int = 4           — causal convolution kernel size
#     mamba_expand: int = 2           — expansion factor
#
#   MoE (Mixture of Experts):
#     n_routed_experts: int = 128     — number of routed expert MLPs
#     n_shared_experts: int = 1       — number of always-active shared experts
#     moe_intermediate_size: int = 1856
#     moe_shared_expert_intermediate_size: int = 3712  — shared expert intermediate dim
#     num_experts_per_tok: int = 6    — top-k experts selected per token
#     routed_scaling_factor: float = 2.5
#
# TODO: Add a @property called layers_block_type that returns a list of strings
#   mapping each character in hybrid_override_pattern to its layer type name:
#     "M" -> "mamba", "*" -> "attention", "-" -> "mlp", "E" -> "moe"
#   (use a list comprehension over range(num_hidden_layers))
#   Mapping dict: {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}

@dataclass
class NemotronHConfig:
    # TODO: Define all fields listed above with their default values,
    #       then add the layers_block_type property
    #
    # Step 1 — General:
    #   vocab_size: int = ...           (131072)
    #   hidden_size: int = ...          (2688)
    #   intermediate_size: int = ...    (1856)
    #   num_hidden_layers: int = ...    (52)
    #   layer_norm_epsilon: float = ... (1e-5)
    #   tie_word_embeddings: bool = ... (False)
    #
    # Step 2 — Hybrid layer pattern:
    #   hybrid_override_pattern: str = ...  ("MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME")
    #
    # Step 3 — Attention:
    #   num_attention_heads: int = ...      (32)
    #   head_dim: int = ...                 (128)
    #   num_key_value_heads: int = ...      (2)
    #   attention_bias: bool = ...          (False)
    #   max_position_embeddings: int = ...  (262144)
    #   rope_theta: float = ...             (10000.0)
    #
    # Step 4 — MLP:
    #   mlp_bias: bool = ...  (False)
    #
    # Step 5 — Mamba2:
    #   mamba_num_heads: int = ...   (64)
    #   mamba_head_dim: int = ...    (64)
    #   mamba_n_groups: int = ...    (8)
    #   ssm_state_size: int = ...    (128)
    #   mamba_d_conv: int = ...      (4)
    #   mamba_expand: int = ...      (2)
    #
    # Step 6 — MoE:
    #   n_routed_experts: int = ...                    (128)
    #   n_shared_experts: int = ...                    (1)
    #   moe_intermediate_size: int = ...               (1856)
    #   moe_shared_expert_intermediate_size: int = ... (3712)
    #   num_experts_per_tok: int = ...                 (6)
    #   routed_scaling_factor: float = ...             (2.5)
    #
    # Step 7 — Property:
    #   @property layers_block_type -> list[str]
    #   (map each char in hybrid_override_pattern: M->mamba, *->attention, - ->mlp, E->moe)
    pass


# Run tests:
# pytest nano-nemotron/01_config_and_primitives/a_nemotron_h_config/test_exercise.py -v
