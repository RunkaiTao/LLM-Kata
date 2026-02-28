# Batch Data Loader

## Concepts

The batch data loader handles preparing training data for the model:

- **Tensor conversion**: Convert encoded text (list of ints) into a `torch.long` tensor
- **Train/val split**: Split data into training and validation sets (e.g., 90/10)
- **`get_batch()`**: Randomly sample batches of input-target pairs
  - Randomly pick `batch_size` starting indices
  - For each index `i`: input `x = data[i : i+block_size]`, target `y = data[i+1 : i+block_size+1]`
  - The target is the input shifted by one position (next-token prediction)
- **Batch dimensions**: Both `x` and `y` have shape `(batch_size, block_size)`

## Your Task

Implement two functions in `exercise.py`:

1. **`prepare_data(encoded_text, train_fraction)`** - Convert to tensor and split into train/val
2. **`get_batch(data, block_size, batch_size, device)`** - Sample a random batch

## Verify

```bash
pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v

# Test individual functions:
pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v -k TestPrepareData
pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v -k TestGetBatch
```
