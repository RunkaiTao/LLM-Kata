# Single-Head Self-Attention

## Concepts

Self-attention allows every position in a sequence to attend to all previous positions (and itself):

- **Key, Query, Value**: Three separate linear projections (no bias) from input
- **Scaled dot-product attention**: `wei = (Q @ K^T) / sqrt(head_size)`
- **Causal masking**: A lower-triangular mask prevents attending to future tokens.
  Positions where the mask is 0 are filled with `-inf` before softmax
- **Softmax** turns scores into attention weights (sum to 1 per row)
- **Weighted aggregation**: `out = wei @ V`
- **`register_buffer`**: The triangular mask is not a learnable parameter but needs
  to be moved to the correct device
- **Dropout** is applied on the attention weights

## Your Task

Implement the `Head` class in `exercise.py`:

1. **`__init__`** - Create key, query, value linear projections, the causal mask buffer, and dropout
2. **`forward(x)`** - Compute scaled dot-product attention with causal masking

## Verify

```bash
pytest nano-gpt/02_layers/b_self_attention/test_exercise.py -v
```
