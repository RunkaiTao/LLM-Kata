# The Training Loop

## Concepts

The training loop is the core of model optimization:

- **AdamW optimizer**: `torch.optim.AdamW(model.parameters(), lr=learning_rate)`
- **Training loop pattern** (for each iteration):
  1. Sample a batch: `xb, yb = get_batch(train_data, ...)`
  2. Forward pass: `logits, loss = model(xb, yb)`
  3. Zero gradients: `optimizer.zero_grad(set_to_none=True)`
  4. Backward pass: `loss.backward()`
  5. Update weights: `optimizer.step()`
- **Periodic evaluation**: Every `eval_interval` steps, call `estimate_loss()` and record it
- The **loss should decrease** over iterations as the model learns

## Your Task

Implement `train_model()` in `exercise.py`. The full model, `get_batch`, and `estimate_loss` are provided.

## Verify

```bash
pytest nano-gpt/05_train/b_training_loop/test_exercise.py -v
```
