# Transformer Block

## Concepts

A transformer block combines self-attention and feed-forward layers with:

- **Pre-LayerNorm architecture**: LayerNorm is applied *before* each sub-layer, not after
- **Residual connections**: The input is added to the output of each sub-layer
- The pattern is:
  ```
  x = x + MultiHeadAttention(LayerNorm(x))   # communication
  x = x + FeedForward(LayerNorm(x))           # computation
  ```
- **LayerNorm** normalizes across the embedding dimension, stabilizing training
- **Residual connections** help gradients flow through deep networks

## Your Task

Implement the `Block` class in `exercise.py`:

1. **`__init__`** - Create self-attention (`sa`), feed-forward (`ffwd`), and two LayerNorm layers (`ln1`, `ln2`)
2. **`forward(x)`** - Apply pre-norm residual connections

The `Head`, `MultiHeadAttention`, and `FeedForward` classes are provided.

## Verify

```bash
pytest nano-gpt/02_layers/e_transformer_block/test_exercise.py -v
```
