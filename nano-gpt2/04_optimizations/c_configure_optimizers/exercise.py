"""
Optimizer configuration with differential weight decay.

GPT-2 training applies weight decay (L2 regularization) only to 2D parameters
(weight matrices in Linear and Embedding layers), NOT to 1D parameters
(biases, LayerNorm scales/shifts). This prevents regularizing parameters
that shouldn't be penalized.

The optimizer also uses specific hyperparameters:
- betas=(0.9, 0.95) for Adam momentum
- eps=1e-8 for numerical stability
- fused=True when available on CUDA (faster kernel)

Reference: Karpathy's build-nanogpt train_gpt2.py lines 179-202
           (commit 3a148e4: "Add weight decay, only for 2D params, and add fused AdamW")
"""
import inspect
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import completed exercises: GPT, GPTConfig
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement configure_optimizers
# ---------------------------------------------------------------------------
def configure_optimizers(model, weight_decay: float, learning_rate: float, device_type: str):
    """
    Create an AdamW optimizer with differential weight decay.

    Args:
        model: A GPT model instance.
        weight_decay: Weight decay coefficient for 2D parameters.
        learning_rate: Learning rate.
        device_type: 'cuda' or 'cpu' — determines whether fused AdamW is used.

    Returns:
        A torch.optim.AdamW optimizer with two parameter groups.

    Steps:
    1. Collect all parameters that require grad:
       param_dict = {pn: p for pn, p in model.named_parameters()}
       param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    2. Separate into decay and no-decay groups:
       decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
         — weight matrices in Linear layers and Embedding layers
       nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
         — biases (1D) and LayerNorm parameters (1D)

    3. Create optimizer parameter groups:
       optim_groups = [
           {'params': decay_params, 'weight_decay': weight_decay},
           {'params': nodecay_params, 'weight_decay': 0.0},
       ]

    4. Check if fused AdamW is available:
       fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
       use_fused = fused_available and device_type == 'cuda'

    5. Create and return the optimizer:
       torch.optim.AdamW(optim_groups, lr=learning_rate,
                         betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    """
    # TODO: Implement configure_optimizers following the steps above
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )
    return optimizer
