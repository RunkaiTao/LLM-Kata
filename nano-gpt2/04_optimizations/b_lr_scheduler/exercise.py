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
    1. Warmup region (it < warmup_steps):
         Linearly scale from ~0 to max_lr
         Use (it + 1) so step 0 gets a small non-zero lr
    2. Post-training region (it > max_steps):
         Return min_lr (constant)
    3. Cosine decay region (between warmup_steps and max_steps):
         Compute a decay_ratio from 0.0 to 1.0 over this range
         Apply cosine schedule to interpolate between max_lr and min_lr
         (use math.cos with math.pi)
    """
    # TODO: Implement get_lr following the steps above
    # Step 1: if it < warmup_steps:
    #             return ...  (linear warmup: max_lr * (it + 1) / warmup_steps)
    # Step 2: if it > max_steps:
    #             return min_lr
    # Step 3: decay_ratio = ...  (0.0 to 1.0 over the cosine decay range)
    #         coeff = ...        (0.5 * (1.0 + math.cos(math.pi * decay_ratio)))
    #         return min_lr + coeff * (max_lr - min_lr)
    pass

# Run tests: pytest nano-gpt2/04_optimizations/b_lr_scheduler/test_exercise.py -v
