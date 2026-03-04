"""
Batch data loader for nano-GPT.

Handles splitting data into train/val and sampling random batches
for training.
"""
import torch


def prepare_data(
    encoded_text: list[int], train_fraction: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert encoded text to a tensor and split into train/val sets.

    Steps:
    1. Convert encoded_text to a torch.long tensor.
    2. Compute the split index n from train_fraction and the data length.
    3. Split data into train_data (first n elements) and val_data (remaining elements).

    Args:
        encoded_text: List of integer token IDs.
        train_fraction: Fraction of data to use for training (default 0.9).

    Returns:
        A tuple of (train_data, val_data), both torch.long tensors.
    """
    # TODO: Implement prepare_data following the steps above
    # Step 1: data = ...        (torch.tensor(encoded_text, dtype=torch.long))
    # Step 2: n = ...           (int(train_fraction * len(data)))
    # Step 3: train_data, val_data = ...  (data[:n], data[n:])
    # return (train_data, val_data)
    pass


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of input-target pairs.

    Steps:
    1. Generate batch_size random starting indices in
       range [0, len(data) - block_size).
    2. For each index i, extract x[i] as a block_size-length window starting at i.
    3. For each index i, extract y[i] as the next-token-shifted window starting at i+1.
    4. Stack into tensors of shape (batch_size, block_size).
    5. Move x, y to the specified device.

    Args:
        data: The data tensor to sample from.
        block_size: The context window size.
        batch_size: Number of sequences per batch.
        device: Device to place tensors on ('cpu' or 'cuda').

    Returns:
        A tuple (x, y) where both are (batch_size, block_size) tensors.
    """
    # TODO: Implement get_batch following the steps above
    # Step 1: ix = ...   (torch.randint(len(data) - block_size, (batch_size,)))
    # Step 2: x = ...    (torch.stack([data[i:i+block_size] for i in ix]))
    # Step 3: y = ...    (torch.stack([data[i+1:i+1+block_size] for i in ix]))
    # Step 4: x, y = ...  (x.to(device), y.to(device))
    # return (x, y)
    pass

# Run tests: pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v
# Test individual functions:
# pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v -k TestPrepareData
# pytest nano-gpt/01_data_and_input/b_batch_loader/test_exercise.py -v -k TestGetBatch
