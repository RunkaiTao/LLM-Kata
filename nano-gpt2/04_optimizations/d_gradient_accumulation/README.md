# Gradient Accumulation

## Concepts

- **Simulating larger batches**: When GPU memory can't fit a large batch, split it into `grad_accum_steps` micro-batches. Run forward+backward on each, then do a single optimizer step. The result is mathematically equivalent to a larger batch.
- **Loss scaling**: Divide loss by `grad_accum_steps` before `backward()`. Since gradients add across `backward()` calls, dividing the loss ensures accumulated gradients equal the mean (not the sum).
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` caps the total gradient norm at 1.0. This prevents exploding gradients that can destabilize training.
- **Detached loss tracking**: Use `loss.detach().item()` to record the loss value without keeping the computation graph in memory.

## Your Task

Implement `train_step()` in `exercise.py`:

1. Zero gradients
2. Loop over `grad_accum_steps` micro-batches: forward, scale loss, accumulate loss value, backward
3. Clip gradient norm to 1.0
4. Optimizer step
5. Return accumulated loss

## Verify

```bash
pytest nano-gpt2/04_optimizations/d_gradient_accumulation/test_exercise.py -v
```
