# Feed-Forward Network

## Concepts

The feed-forward network (FFN) in a transformer block:

- **Expands** the representation then **compresses** it back
- Two linear layers: `n_embd -> 4*n_embd -> n_embd`
- The inner expansion factor of **4x** is standard from the original Transformer paper
- **ReLU** activation between the two linear layers
- **Dropout** after the second linear layer
- **`nn.Sequential`** makes it easy to chain layers
- The FFN is applied independently to each position (same weights, no cross-position interaction)

## Your Task

Implement the `FeedForward` class in `exercise.py`:

1. **`__init__`** - Create an `nn.Sequential` with Linear -> ReLU -> Linear -> Dropout
2. **`forward(x)`** - Pass input through the sequential network

## Verify

```bash
pytest nano-gpt/02_layers/d_feed_forward/test_exercise.py -v
```
