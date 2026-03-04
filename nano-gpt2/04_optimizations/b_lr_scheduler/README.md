# Learning Rate Scheduler

## Concepts

- **Linear warmup**: The learning rate starts near zero and increases linearly to `max_lr` over `warmup_steps`. This prevents large, unstable updates early in training when the model's gradients are noisy.
- **Cosine decay**: After warmup, the learning rate smoothly decreases following a cosine curve from `max_lr` to `min_lr`. The formula: `min_lr + 0.5 * (1 + cos(pi * decay_ratio)) * (max_lr - min_lr)`.
- **Post-training constant**: After `max_steps`, the learning rate stays at `min_lr`.
- **Pure function**: This is just math — no PyTorch modules or state. Given a step number, return the learning rate.

## Your Task

Implement `get_lr()` in `exercise.py`:

1. If `it < warmup_steps`: return linear warmup value (`max_lr * (it+1) / warmup_steps`)
2. If `it > max_steps`: return `min_lr`
3. Otherwise: return cosine decay value

## Verify

```bash
pytest nano-gpt2/04_optimizations/b_lr_scheduler/test_exercise.py -v
```
