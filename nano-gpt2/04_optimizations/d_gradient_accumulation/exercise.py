"""
Gradient accumulation for simulating larger batch sizes.

When GPU memory is limited, we can't fit a large batch in one forward pass.
Instead, we split the large batch into "micro-batches", run forward+backward
on each (accumulating gradients), then do a single optimizer step.

Key insight: loss must be divided by grad_accum_steps before backward(),
because gradients ADD across backward() calls. We want the MEAN gradient,
not the SUM.

This exercise also introduces gradient clipping (clip_grad_norm_) which
prevents exploding gradients by capping the total gradient norm.

Reference: Karpathy's build-nanogpt train_gpt2.py lines 484-503
           (commit 01be6b3: "Add gradient accumulation")
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
# YOUR TASK: Implement train_step with gradient accumulation
# ---------------------------------------------------------------------------
def train_step(model, optimizer, train_loader, grad_accum_steps: int, device: str = "cpu"):
    """
    Perform one full training step with gradient accumulation and gradient clipping.

    Args:
        model: A GPT model instance (already on device).
        optimizer: The optimizer (already configured).
        train_loader: A DataLoaderLite instance.
        grad_accum_steps: Number of micro-batches to accumulate over.
        device: Device string.

    Returns:
        The accumulated loss value (float) for this training step.

    Steps:
    1. optimizer.zero_grad()    — clear gradients from previous step
    2. loss_accum = 0.0
    3. For micro_step in range(grad_accum_steps):
       a. x, y = train_loader.next_batch()
       b. x, y = x.to(device), y.to(device)
       c. logits, loss = model(x, y)
       d. loss = loss / grad_accum_steps
          — scale loss so accumulated gradients equal the mean
       e. loss_accum += loss.detach().item()
          — track total loss (detach to avoid keeping graph)
       f. loss.backward()
          — gradients ACCUMULATE (they add up across calls)
    4. norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       — clip gradient norm to max 1.0 to prevent explosions
    5. optimizer.step()
    6. Return loss_accum
    """
    # TODO: Implement train_step following the steps above
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach().item()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_accum
