"""
Validation loss estimation for monitoring training progress.

During training, we periodically evaluate the model on held-out validation
data to detect overfitting. This function:
1. Switches the model to eval mode (disables dropout)
2. Resets the validation loader to the start
3. Averages loss over multiple batches for a smooth estimate
4. Restores the model to training mode

Reference: Karpathy's build-nanogpt train_gpt2.py lines 381-398
           (commit 21d3d32: "Add validation split")
"""
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import completed exercises: GPT, GPTConfig, DataLoaderLite
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig
DataLoaderLite = load("02_data_and_tokenization", "b_data_loader_lite").DataLoaderLite


# ---------------------------------------------------------------------------
# YOUR TASK: Implement estimate_val_loss
# ---------------------------------------------------------------------------
def estimate_val_loss(model, val_loader, val_loss_steps: int, device: str = "cpu"):
    """
    Estimate validation loss by averaging over multiple batches.

    Args:
        model: A GPT model instance.
        val_loader: A DataLoaderLite for validation data.
        val_loss_steps: Number of batches to average over.
        device: Device string.

    Returns:
        The average validation loss as a float.

    Steps:
    1. model.eval()              — switch to evaluation mode (disables dropout)
    2. val_loader.reset()        — start from the beginning of validation data
    3. val_loss_accum = 0.0
    4. with torch.no_grad():     — disable gradient computation for efficiency
    5.   For _ in range(val_loss_steps):
           a. x, y = val_loader.next_batch()
           b. x, y = x.to(device), y.to(device)
           c. logits, loss = model(x, y)
           d. val_loss_accum += loss.item() / val_loss_steps
              — divide by val_loss_steps to get the running mean
    6. model.train()             — restore training mode
    7. Return val_loss_accum
    """
    # TODO: Implement estimate_val_loss following the steps above
    model.eval()
    val_loader.reset()
    val_loss_accum = 0.0
    with torch.no_grad():
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            val_loss_accum += loss.item() / val_loss_steps
    model.train()
    return val_loss_accum
