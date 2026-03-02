"""
Learning rate scheduler with linear warmup and cosine decay.

GPT-2 training uses a learning rate schedule that:
1. Linearly warms up from near-zero to max_lr over warmup_steps
2. Cosine-decays from max_lr down to min_lr over the remaining steps
3. Stays at min_lr after max_steps

This is a pure function (no PyTorch modules) — just math.

Reference: Karpathy's build-nanogpt train_gpt2.py lines 353-364
           (commit 90e5d15: "Add learning rate scheduler")
"""
import math


# ---------------------------------------------------------------------------
# YOUR TASK: Implement get_lr
# ---------------------------------------------------------------------------
def get_lr(it: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Compute the learning rate for a given training iteration.

    Args:
        it: Current training iteration (0-indexed).
        warmup_steps: Number of linear warmup steps.
        max_steps: Total number of training steps (for cosine decay endpoint).
        max_lr: Maximum (peak) learning rate.
        min_lr: Minimum learning rate after decay.

    Returns:
        The learning rate for iteration `it`.

    Steps:
    1. If it < warmup_steps:
         return max_lr * (it + 1) / warmup_steps
         — linear warmup from ~0 to max_lr
         — note: (it + 1) so that step 0 gets a small non-zero lr
    2. If it > max_steps:
         return min_lr
         — constant after training is done
    3. Otherwise (cosine decay region):
         decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
         coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
         — coeff goes from 1.0 (at warmup_steps) to 0.0 (at max_steps)
         return min_lr + coeff * (max_lr - min_lr)
    """
    # TODO: Implement get_lr following the steps above
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
