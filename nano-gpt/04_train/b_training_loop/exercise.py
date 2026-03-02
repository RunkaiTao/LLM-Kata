"""
Training loop for nano-GPT.
"""
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Import completed exercises: GPTLanguageModel, get_batch, estimate_loss
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPTLanguageModel = load("03_combine_layers", "a_assemble_model").GPTLanguageModel
get_batch = load("01_data_and_input", "b_batch_loader").get_batch
estimate_loss = load("04_train", "a_loss_estimation").estimate_loss


# ---------------------------------------------------------------------------
# YOUR TASK: Implement train_model
# ---------------------------------------------------------------------------
def train_model(
    model,
    train_data,
    val_data,
    block_size,
    batch_size,
    max_iters,
    learning_rate,
    eval_interval,
    eval_iters,
    device="cpu",
):
    """
    Train the GPT model.

    Args:
        model: The GPTLanguageModel instance (already on device).
        train_data: Training data tensor.
        val_data: Validation data tensor.
        block_size: Context window size.
        batch_size: Training batch size.
        max_iters: Number of training iterations.
        learning_rate: Learning rate for optimizer.
        eval_interval: How often to estimate loss.
        eval_iters: Number of batches for loss estimation.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        A list of loss dicts recorded at each eval point.

    Steps:
    1. Create an AdamW optimizer over model.parameters() with the given learning_rate
       (use torch.optim.AdamW).
    2. Initialize an empty list to record loss evaluations.
    3. For iter in range(max_iters):
       a. If iter is a multiple of eval_interval, or it's the last iteration:
          call estimate_loss and append the result to the list.
       b. Sample a batch (xb, yb) from train_data using get_batch.
       c. Forward pass: get logits and loss from the model.
       d. Zero the gradients (use set_to_none=True for efficiency).
       e. Backpropagate the loss.
       f. Step the optimizer.
    4. Return the list of recorded losses.
    """
    # TODO: Implement the training loop
    raise NotImplementedError("Implement train_model")

# Run tests: pytest nano-gpt/04_train/b_training_loop/test_exercise.py -v
