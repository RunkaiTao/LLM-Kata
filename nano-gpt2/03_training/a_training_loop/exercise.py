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
    1. Create an AdamW optimizer: torch.optim.AdamW(model.parameters(), lr=learning_rate)
    2. Initialize an empty list loss_history to record loss at each step.
    3. For step in range(max_steps):
       a. x, y = train_loader.next_batch()
       b. x, y = x.to(device), y.to(device)
       c. logits, loss = model(x, y)    — forward pass with targets computes loss
       d. optimizer.zero_grad(set_to_none=True)  — clear old gradients efficiently
       e. loss.backward()                — backpropagate
       f. optimizer.step()               — update parameters
       g. loss_history.append(loss.item())  — record the scalar loss value
    4. Return loss_history
    """
    # TODO: Implement train following the steps above
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history = []
    for step in range(max_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return loss_history
