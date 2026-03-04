# Data Loader Lite

## Concepts

- **Sequential loading**: Unlike nano-gpt (which randomly samples batches), GPT-2 training reads tokens sequentially. A position pointer advances through the data after each batch.
- **Next-token prediction**: For each batch, inputs `x` and targets `y` are consecutive sequences shifted by 1. If `buf` is a contiguous slice of tokens, then `x = buf[:-1]` and `y = buf[1:]`.
- **Shards**: Large datasets are split into multiple files ("shards"). When the current shard is exhausted, the loader rotates to the next shard and resets the position.
- **Buffer size**: Each batch needs `B * T + 1` tokens — the extra token provides the final target.

## Your Task

Implement `DataLoaderLite` in `exercise.py`:

1. `__init__`: Store shards, batch size `B`, sequence length `T`, and call `reset()`
2. `reset()`: Set position to the beginning of the first shard
3. `next_batch()`: Extract sequential `(x, y)` tensors of shape `(B, T)`, advance the position pointer, and rotate to the next shard when exhausted

## Verify

```bash
pytest nano-gpt2/02_data_and_tokenization/b_data_loader_lite/test_exercise.py -v
```
