# Training Loop

## Concepts

- **AdamW optimizer**: The standard optimizer for transformer training. AdamW decouples weight decay from the gradient update, unlike the original Adam.
- **Training step pattern**: Each step follows: get batch -> forward pass (compute loss) -> zero gradients -> backward pass -> optimizer step.
- **`zero_grad(set_to_none=True)`**: More memory-efficient than `set_to_none=False` — sets gradients to `None` instead of zero tensors.
- **Loss tracking**: Record `loss.item()` at each step to monitor training progress. Loss should decrease over time.

## Your Task

Implement `train()` in `exercise.py`:

1. Create an AdamW optimizer
2. For each step: get a batch from the loader, forward pass with targets, zero gradients, backward, optimizer step, record loss
3. Return the list of loss values

Note: The model, `GPTConfig`, and `DataLoaderLite` are provided via imports. No gradient accumulation or LR scheduling yet — those come in later exercises.

## Verify

```bash
pytest nano-gpt2/03_training/a_training_loop/test_exercise.py -v
```
