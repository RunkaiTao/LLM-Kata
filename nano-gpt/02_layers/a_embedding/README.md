# Token and Position Embeddings

## Concepts

Embeddings convert discrete token IDs into continuous vector representations:

- **`nn.Embedding`**: A lookup table that maps integer indices to dense vectors
- **Token embeddings**: Encode *what* the token is. Shape: `(vocab_size, n_embd)`
- **Position embeddings**: Encode *where* the token is in the sequence. Shape: `(block_size, n_embd)`
- The two are **summed element-wise**: `x = token_emb + position_emb`
- Position indices are created with `torch.arange(T, device=idx.device)`
- **Broadcasting**: Position embedding `(T, C)` broadcasts across batch dimension `(B, T, C)`

## Your Task

Implement the `Embeddings` module in `exercise.py`:

1. **`__init__`** - Create two `nn.Embedding` layers (token and position)
2. **`forward(idx)`** - Look up both embeddings and sum them

## Verify

```bash
pytest nano-gpt/02_layers/a_embedding/test_exercise.py -v
```
