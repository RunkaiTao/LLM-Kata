# Multi-Head Attention

## Concepts

Multi-head attention runs several attention heads in parallel:

- Each head operates on a subset of the embedding dimensions
- `head_size = n_embd // n_head` (e.g., 384 // 6 = 64)
- Outputs from all heads are **concatenated** along the last dimension:
  `(B, T, head_size * n_head) = (B, T, n_embd)`
- A **linear projection** maps the concatenated output back to `n_embd`
- **Dropout** is applied after the projection
- **`nn.ModuleList`** is used so PyTorch properly tracks the sub-modules

## Your Task

Implement the `MultiHeadAttention` class in `exercise.py`:

1. **`__init__`** - Create a ModuleList of `Head` instances, a projection layer, and dropout
2. **`forward(x)`** - Run each head, concatenate outputs, project, apply dropout

The `Head` class is provided for you.

## Verify

```bash
pytest nano-gpt/02_layers/c_multi_head_attention/test_exercise.py -v
```
