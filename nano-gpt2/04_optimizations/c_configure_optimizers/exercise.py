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
    1. Collect all parameters that require grad into a dict
       (use model.named_parameters, filter by requires_grad)

    2. Separate into two groups based on dimensionality:
       - decay_params: 2D+ parameters (weight matrices) — apply weight decay
       - nodecay_params: <2D parameters (biases, LayerNorm) — no weight decay
       (use p.dim() to distinguish)

    3. Create two optimizer parameter groups with appropriate weight_decay settings

    4. Check if fused AdamW is available by inspecting the AdamW signature
       for a 'fused' parameter; use it only on CUDA (use inspect.signature)

    5. Create and return an AdamW optimizer with betas=(0.9, 0.95), eps=1e-8
    """
    # TODO: Implement configure_optimizers following the steps above
    # Step 1: param_dict = ...      (dict of named params that require grad)
    # Step 2: decay_params = ...    (list of params with dim >= 2)
    #         nodecay_params = ...  (list of params with dim < 2)
    # Step 3: optim_groups = [
    #             {"params": decay_params, "weight_decay": weight_decay},
    #             {"params": nodecay_params, "weight_decay": 0.0},
    #         ]
    # Step 4: fused_available = ... (check inspect.signature for 'fused' param)
    #         use_fused = ...       (fused_available and device_type == "cuda")
    # Step 5: optimizer = ...       (AdamW with betas=(0.9, 0.95), eps=1e-8)
    # return optimizer
    pass

# Run tests: pytest nano-gpt2/04_optimizations/c_configure_optimizers/test_exercise.py -v
