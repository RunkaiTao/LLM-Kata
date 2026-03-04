"""
Basic training loop for GPT-2.

This exercise implements a simple training loop: forward pass, compute loss,
backward pass, optimizer step. No gradient accumulation or learning rate
scheduling yet — those come in the optimizations section.

Reference: Karpathy's build-nanogpt train_gpt2.py lines 482-508 (simplified)
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
# YOUR TASK: Implement train
# ---------------------------------------------------------------------------
def train(model, train_loader, max_steps, learning_rate, device="cpu"):
    """
    Train the GPT model for max_steps steps.

    Args:
        model: A GPT model instance (already on device).
        train_loader: A DataLoaderLite instance providing batches.
        max_steps: Number of training iterations.
        learning_rate: Learning rate for AdamW optimizer.
        device: Device to use ('cpu' or 'cuda').

    Returns:
        A list of float loss values, one per training step.

    Steps:
    1. Create an AdamW optimizer for the model (use torch.optim.AdamW)
    2. Initialize an empty loss_history list
    3. For each of max_steps iterations:
       a. Get the next batch from train_loader and move to device
       b. Forward pass with targets to get logits and loss
       c. Zero gradients, backpropagate, and step the optimizer
          (use optimizer.zero_grad with set_to_none=True for efficiency)
       d. Record the scalar loss value in loss_history
    4. Return loss_history
    """
    # TODO: Implement train following the steps above
    # Step 1: optimizer = ...    (torch.optim.AdamW)
    # Step 2: loss_history = []
    # Step 3: for step in range(max_steps):
    #     a. x, y = ...          (next batch, move to device)
    #     b. logits, loss = ...  (forward pass with targets)
    #     c. optimizer.zero_grad(set_to_none=True)
    #        loss.backward()
    #        optimizer.step()
    #     d. loss_history.append(loss.item())
    # return loss_history
    pass

# Run tests: pytest nano-gpt2/03_training/a_training_loop/test_exercise.py -v
