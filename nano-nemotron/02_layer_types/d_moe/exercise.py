"""
Mixture of Experts (MoE) Layer for NemotronH.

NemotronH uses sparse MoE layers (marked "E" in the hybrid pattern) where each
token is routed to only a subset of expert networks. This allows scaling model
capacity without proportionally increasing compute per token.

Key design choices in NemotronH MoE:
- Non-gated MoE: each expert is a simple MLP (up_proj -> activation -> down_proj),
  NOT a gated MLP like in LLaMA/Mixtral
- Sigmoid routing: the router uses sigmoid (not softmax) to score experts
- Top-k selection: each token selects the top-k experts (k=6 by default)
- Shared experts: one expert always processes all tokens (not routed)
- Combined output: routed_output * scaling_factor + shared_output

Architecture:
    1. Router: linear(hidden_size -> n_routed_experts) with sigmoid scoring
    2. Top-k selection of experts per token
    3. Shared expert: always-active MLP processing all tokens
    4. Routed experts: only selected experts process each token
    5. Combine: scaled routed output + shared output

Reference: vllm/vllm/model_executor/models/nemotron_h.py lines 127-279
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
NemotronHMLP = load("02_layer_types", "a_mlp").NemotronHMLP


# ---------------------------------------------------------------------------
# YOUR TASK: Implement NemotronHMoE
# ---------------------------------------------------------------------------

class NemotronHMoE(nn.Module):
    """Sparse Mixture of Experts with shared expert."""

    def __init__(self, config: "NemotronHConfig"):
        """
        Args:
            config: NemotronHConfig instance.

        Steps:
        1. Store config values:
           - self.hidden_size = config.hidden_size
           - self.n_routed_experts = config.n_routed_experts             (128)
           - self.n_shared_experts = config.n_shared_experts             (1)
           - self.num_experts_per_tok = config.num_experts_per_tok       (6)
           - self.routed_scaling_factor = config.routed_scaling_factor   (2.5)
        2. Create self.gate as nn.Linear(hidden_size, n_routed_experts, bias=False)
        3. Create self.shared_experts as a NemotronHMLP instance:
           NemotronHMLP(config, hidden_size=hidden_size,
                        intermediate_size=config.moe_shared_expert_intermediate_size)
        4. Create self.experts as nn.ModuleList of n_routed_experts NemotronHMLP instances:
           Each expert: NemotronHMLP(config, hidden_size=hidden_size,
                                     intermediate_size=config.moe_intermediate_size)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.hidden_size = ..., self.n_routed_experts = ..., self.n_shared_experts = ...
        #         self.num_experts_per_tok = ..., self.routed_scaling_factor = ...
        # Step 2: self.gate = ...            (nn.Linear: hidden_size -> n_routed_experts, no bias)
        # Step 3: self.shared_experts = ...  (NemotronHMLP with moe_shared_expert_intermediate_size)
        # Step 4: self.experts = nn.ModuleList(...)  (n_routed_experts NemotronHMLP instances with moe_intermediate_size)
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input of shape (..., hidden_size).

        Returns:
            Output of shape (..., hidden_size).

        Steps:
        1. Save original shape and flatten to 2D: (num_tokens, hidden_size)
           (use .view(-1, hidden_dim))
        2. Compute router logits: router_logits = self.gate(hidden_states)
        3. Compute routing weights using sigmoid (NOT softmax)
        4. Select top-k experts per token (use torch.topk)
        5. Normalize top-k weights so they sum to 1 per token
        6. Compute routed expert outputs:
           a. Initialize routed_output = zeros_like(hidden_states)
           b. For each expert_idx in range(n_routed_experts):
              i.   Create token_mask: (top_indices == expert_idx).any(dim=-1)
              ii.  If no tokens routed, skip (continue)
              iii. Get token_indices from mask
              iv.  Compute routing weights for these tokens
              v.   Run expert forward on selected tokens
              vi.  Scale by weights and accumulate into routed_output
        7. Scale routed output by self.routed_scaling_factor
        8. Compute shared expert output: shared_output = self.shared_experts(hidden_states)
        9. Combine: final_output = routed_output + shared_output
        10. Reshape back to original shape and return
        """
        # TODO: Implement forward following the steps above
        # Step 1:  original_shape = ...; hidden_states = ...  (flatten to 2D: (num_tokens, hidden_size))
        # Step 2:  router_logits = ...    (self.gate(hidden_states))
        # Step 3:  routing_weights = ...  (torch.sigmoid)
        # Step 4:  top_weights, top_indices = ...  (torch.topk with num_experts_per_tok)
        # Step 5:  top_weights = ...      (normalize: / top_weights.sum(dim=-1, keepdim=True))
        # Step 6:  routed_output = ...    (loop over experts: mask tokens, run expert, accumulate weighted output)
        # Step 7:  routed_output = ...    (scale by routed_scaling_factor)
        # Step 8:  shared_output = ...    (self.shared_experts(hidden_states))
        # Step 9:  final_output = ...     (routed_output + shared_output)
        # Step 10: return ...             (reshape to original_shape)
        pass


# Run tests:
# pytest nano-nemotron/02_layer_types/d_moe/test_exercise.py -v
