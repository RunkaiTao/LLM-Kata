# Validation Loss

## Concepts

- **Monitoring overfitting**: Periodically evaluate on held-out validation data. If training loss keeps dropping but validation loss increases, the model is overfitting.
- **Eval mode**: `model.eval()` disables dropout (if present) and changes BatchNorm behavior. Always switch back with `model.train()` afterward.
- **torch.no_grad()**: Disables gradient computation during evaluation, saving memory and computation. Gradients aren't needed since we're not updating parameters.
- **Loader reset**: Call `val_loader.reset()` to start from the beginning of validation data each time, ensuring consistent evaluation.

## Your Task

Implement `estimate_val_loss()` in `exercise.py`:

1. Switch to eval mode
2. Reset the validation loader
3. Average loss over `val_loss_steps` batches under `torch.no_grad()`
4. Restore training mode
5. Return the average validation loss

## Verify

```bash
pytest nano-gpt2/05_evaluation/a_validation_loss/test_exercise.py -v
```
