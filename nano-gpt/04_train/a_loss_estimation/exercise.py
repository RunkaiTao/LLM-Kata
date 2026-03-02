"""
Loss estimation for monitoring training progress.
"""
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import completed exercises: GPTLanguageModel from 03, get_batch from 01/b
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTLanguageModel = load("03_combine_layers", "a_assemble_model").GPTLanguageModel
get_batch = load("01_data_and_input", "b_batch_loader").get_batch


# ---------------------------------------------------------------------------
# YOUR TASK: Implement estimate_loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device="cpu"):
    """
    Estimate the average loss on train and val splits.

    Args:
        model: The GPT model.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        block_size: Context window size.
        batch_size: Batch size for evaluation.
        eval_iters: Number of batches to average over.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        A dict {'train': mean_train_loss, 'val': mean_val_loss}

    Steps:
    1. Initialize an empty output dict.
    2. Set model to evaluation mode (use model.eval()).
    3. For each split in ['train', 'val']:
       a. Select the appropriate data tensor based on the split name.
       b. Create a zeros tensor of shape (eval_iters,) to store per-iteration losses.
       c. For k in range(eval_iters):
          - Get a batch (X, Y) using get_batch with the selected data.
          - Run the model forward to get logits and loss.
          - Store the scalar loss value in losses[k].
       d. Store the mean of losses in the output dict under the split key.
    4. Set model back to training mode (use model.train()).
    5. Return the output dict.
    """
    # TODO: Implement loss estimation
    raise NotImplementedError("Implement estimate_loss")

# Run tests: pytest nano-gpt/04_train/a_loss_estimation/test_exercise.py -v
