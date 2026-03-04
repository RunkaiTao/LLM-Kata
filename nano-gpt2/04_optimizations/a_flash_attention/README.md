# Flash Attention

## Concepts

- **F.scaled_dot_product_attention**: PyTorch's fused attention kernel that replaces the manual sequence of `scale -> mask -> softmax -> matmul`. A single function call does it all.
- **Memory efficiency**: Manual attention materializes the full `(T, T)` attention matrix, using O(T^2) memory. Flash attention uses a tiling algorithm that keeps memory at O(T).
- **No explicit causal mask**: The `is_causal=True` parameter handles causal masking internally — no need to `register_buffer` a triangular mask. This is the key difference from exercise 01/b.
- **Same interface**: The `__init__` and `forward` signatures are identical to 01/b. Only the internal attention computation changes.

## Your Task

Implement `CausalSelfAttention` in `exercise.py` (same as 01/b but with flash attention):

1. `__init__`: Same as 01/b but **without** `register_buffer` for the causal mask
2. `forward`: Same reshaping, but replace manual attention with `F.scaled_dot_product_attention(q, k, v, is_causal=True)`

## Verify

```bash
pytest nano-gpt2/04_optimizations/a_flash_attention/test_exercise.py -v
```
