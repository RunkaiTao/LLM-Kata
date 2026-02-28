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
    2. Compute the split index: n = int(train_fraction * len(data)).
    3. train_data = data[:n], val_data = data[n:].

    Args:
        encoded_text: List of integer token IDs.
        train_fraction: Fraction of data to use for training (default 0.9).

    Returns:
        A tuple of (train_data, val_data), both torch.long tensors.
    """
    data = torch.tensor(encoded_text, dtype=torch.long)
    n = int(train_fraction * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


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
    2. For each index i, x[i] = data[i : i+block_size].
    3. For each index i, y[i] = data[i+1 : i+block_size+1].
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
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
