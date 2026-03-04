# Configure Optimizers

## Concepts

- **Differential weight decay**: Weight decay (L2 regularization) is applied only to 2D+ parameters (weight matrices in Linear and Embedding layers). 1D parameters (biases, LayerNorm scale/shift) are excluded because regularizing them hurts performance.
- **Parameter groups**: AdamW supports multiple parameter groups with different settings. We create two groups: one with `weight_decay` and one with `weight_decay=0.0`.
- **GPT-2 hyperparameters**: `betas=(0.9, 0.95)` (momentum terms), `eps=1e-8` (numerical stability).
- **Fused AdamW**: On CUDA, PyTorch offers a fused kernel (`fused=True`) that combines the optimizer operations into fewer GPU calls for better performance.

## Your Task

Implement `configure_optimizers()` in `exercise.py`:

1. Collect all parameters that require grad
2. Separate into decay (`dim >= 2`) and no-decay (`dim < 2`) groups
3. Create two parameter groups with appropriate `weight_decay` values
4. Check for fused AdamW availability and create the optimizer with `betas=(0.9, 0.95)`, `eps=1e-8`

## Verify

```bash
pytest nano-gpt2/04_optimizations/c_configure_optimizers/test_exercise.py -v
```
