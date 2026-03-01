# Loss Estimation

## Concepts

`estimate_loss()` evaluates the model on multiple batches and returns the mean loss:

- **`@torch.no_grad()`**: Disables gradient computation for efficiency during evaluation
- **`model.eval()`**: Sets model to evaluation mode (disables dropout, etc.)
- **`model.train()`**: Sets it back to training mode afterward
- Averaging over many batches gives a **smoother, more reliable** loss estimate than a single batch
- Returns a dictionary with `'train'` and `'val'` keys

## Your Task

Implement `estimate_loss()` in `exercise.py`. The full model and `get_batch` function are provided.

## Verify

```bash
pytest nano-gpt/04_train/a_loss_estimation/test_exercise.py -v
```
