"""
Mamba2 State Space Model (SSM) Layer — simplified pure-PyTorch version.

The Mamba2 layer replaces attention with a linear-time recurrence (State Space Model).
While attention has O(n^2) complexity in sequence length, the SSM processes sequences
in O(n) by maintaining a hidden state that is updated at each time step.

The key insight of Mamba (Selective SSM) is that parameters B, C, and dt are
*input-dependent* — they are projected from the input rather than being fixed.
This "selectivity" lets the model decide what to remember and what to forget
based on the current input, making it more expressive than fixed-parameter SSMs.

Architecture overview:
    1. in_proj: project hidden_states -> [gate, x, B, C, dt]
    2. conv1d:  causal 1D convolution on [x, B, C] for local context
    3. SSM:     sequential state-space scan using x, dt, A, B, C, D
    4. gate:    SiLU(gate) * RMSNorm(ssm_output)
    5. out_proj: project back to hidden_size

SSM recurrence (per time step t):
    dt_t = softplus(dt[t] + dt_bias)         # positive time step
    A_bar = exp(A * dt_t)                     # discretized state decay
    h = A_bar * h + dt_t * x[t] outer B[t]   # state update
    y[t] = (C[t] . h).sum + D * x[t]         # output (. is element-wise)

Where:
    A: (num_heads,)              — fixed diagonal state matrix (negative, learned)
    D: (num_heads,)              — skip connection weight (learned)
    dt_bias: (num_heads,)        — time step bias (learned)
    B: (seq, n_groups, state)    — input-dependent input matrix
    C: (seq, n_groups, state)    — input-dependent output matrix
    dt: (seq, num_heads)         — input-dependent time step

Reference: vllm/vllm/model_executor/layers/mamba/mamba_mixer2.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement Mamba2Mixer (simplified pure-PyTorch version)
# ---------------------------------------------------------------------------

class Mamba2Mixer(nn.Module):
    """Simplified Mamba2 SSM layer for educational purposes."""

    def __init__(self, config: "NemotronHConfig"):
        """
        Args:
            config: NemotronHConfig instance.

        Steps:
        1. Store dimensions from config:
           - self.hidden_size = config.hidden_size                              (2688)
           - self.num_heads = config.mamba_num_heads                            (64)
           - self.head_dim = config.mamba_head_dim                              (64)
           - self.ssm_state_size = config.ssm_state_size                       (128)
           - self.n_groups = config.mamba_n_groups                              (8)
           - self.intermediate_size = self.num_heads * self.head_dim            (4096)
           - self.conv_kernel_size = config.mamba_d_conv                        (4)
           - self.groups_ssm_state_size = self.n_groups * self.ssm_state_size   (1024)
        2. Create self.in_proj as nn.Linear:
           input = hidden_size, output = 2*intermediate_size + 2*groups_ssm_state_size + num_heads
           bias=False
        3. Create self.conv1d as nn.Conv1d (depthwise):
           in/out_channels = intermediate_size + 2*groups_ssm_state_size (conv_dim)
           kernel_size = conv_kernel_size, groups = in_channels, padding = conv_kernel_size - 1
           bias=True
        4. Create SSM parameters (nn.Parameter):
           - self.A: empty(num_heads), init with -torch.exp(torch.randn(num_heads))
           - self.D: ones(num_heads)
           - self.dt_bias: ones(num_heads)
        5. Create norm:
           - self.norm_weight as nn.Parameter(torch.ones(intermediate_size))
           - self.norm_eps = config.layer_norm_epsilon
        6. Create self.out_proj as nn.Linear(intermediate_size, hidden_size, bias=False)
        """
        super().__init__()
        # TODO: Implement __init__ following the steps above
        # Step 1: self.hidden_size = ..., self.num_heads = ..., self.head_dim = ...
        #         self.ssm_state_size = ..., self.n_groups = ...
        #         self.intermediate_size = ...  (num_heads * head_dim)
        #         self.conv_kernel_size = ...   (config.mamba_d_conv)
        #         self.groups_ssm_state_size = ...  (n_groups * ssm_state_size)
        # Step 2: self.in_proj = ...    (nn.Linear: hidden_size -> 2*intermediate_size + 2*groups_ssm_state_size + num_heads, no bias)
        # Step 3: self.conv1d = ...     (nn.Conv1d: depthwise, in/out=conv_dim, kernel=conv_kernel_size, groups=conv_dim, padding=conv_kernel_size-1)
        # Step 4: self.A = ..., self.D = ..., self.dt_bias = ...  (nn.Parameter: SSM params)
        # Step 5: self.norm_weight = ..., self.norm_eps = ...     (nn.Parameter(ones(intermediate_size)), config.layer_norm_epsilon)
        # Step 6: self.out_proj = ...   (nn.Linear: intermediate_size -> hidden_size, no bias)
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input of shape (B, T, hidden_size).

        Returns:
            Output of shape (B, T, hidden_size).

        Steps:
        1. Project: projected = self.in_proj(hidden_states)  -> (B, T, total_proj_size)
        2. Split projected into gate, x_B_C, dt along last dim:
           sizes = [intermediate_size, conv_dim, num_heads]
           where conv_dim = intermediate_size + 2 * groups_ssm_state_size
        3. Causal convolution on x_B_C:
           a. Transpose to (B, conv_dim, T)
           b. Apply conv1d -> (B, conv_dim, T + padding)
           c. Slice [..., :T] to remove right-padding
           d. Apply F.silu
           e. Transpose back to (B, T, conv_dim)
        4. Split conv output into x, B_mat, C_mat:
           sizes = [intermediate_size, groups_ssm_state_size, groups_ssm_state_size]
        5. Reshape for SSM:
           x -> (B, T, num_heads, head_dim),
           B_mat -> (B, T, n_groups, ssm_state_size),
           C_mat -> (B, T, n_groups, ssm_state_size)
        6. SSM scan (sequential loop over T time steps):
           a. h = zeros(B, num_heads, head_dim, ssm_state_size)
           b. heads_per_group = num_heads // n_groups
           c. For each t:
              i.   dt_t = F.softplus(dt[:, t] + self.dt_bias)
              ii.  A_bar = exp(self.A * dt_t), reshape to (B, num_heads, 1, 1)
              iii. B_t = B_mat[:, t].repeat_interleave(heads_per_group, dim=1)
              iv.  x_t = x[:, t]
              v.   outer = x_t.unsqueeze(-1) * B_t.unsqueeze(-2), scale by dt_t
              vi.  h = A_bar * h + scaled_outer
              vii. C_t = C_mat[:, t].repeat_interleave(heads_per_group, dim=1)
              viii. y_t = (h * C_t.unsqueeze(-2)).sum(-1) + self.D * x_t
              ix.  Append y_t
           d. y = torch.stack(outputs, dim=1) -> (B, T, num_heads, head_dim)
           e. Reshape to (B, T, intermediate_size)
        7. Gated normalization:
           a. gate_activated = F.silu(gate)
           b. y = y * gate_activated
           c. RMSNorm: variance = y.pow(2).mean(-1, keepdim=True)
              y = y * rsqrt(variance + eps) * self.norm_weight
        8. Output projection: output = self.out_proj(y)
        9. Return output
        """
        # TODO: Implement forward following the steps above
        # Step 1: projected = ...          (self.in_proj(hidden_states))
        # Step 2: gate, x_B_C, dt = ...   (torch.split: [intermediate_size, conv_dim, num_heads])
        # Step 3: x_B_C = ...             (transpose -> conv1d -> slice [:T] -> F.silu -> transpose back)
        # Step 4: x, B_mat, C_mat = ...   (torch.split: [intermediate_size, groups_ssm_state_size, groups_ssm_state_size])
        # Step 5: x, B_mat, C_mat = ...   (reshape for SSM)
        # Step 6: y = ...                 (SSM scan loop: init h=zeros; for t: dt_t, A_bar, B_t, x_t, outer, h update, C_t, y_t; stack)
        # Step 7: y = ...                 (F.silu(gate) * y, then RMSNorm: y * rsqrt(var + eps) * norm_weight)
        # Step 8: output = ...            (self.out_proj(y))
        # return output
        pass


# Run tests:
# pytest nano-nemotron/02_layer_types/c_mamba2/test_exercise.py -v
