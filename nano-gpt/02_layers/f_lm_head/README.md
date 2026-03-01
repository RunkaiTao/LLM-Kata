# Language Model Head

## Concepts

The LM head is the final layer of the GPT model that converts hidden states into vocabulary predictions:

- **Final LayerNorm**: `nn.LayerNorm(n_embd)` stabilizes the transformer output before projection
- **Linear projection**: `nn.Linear(n_embd, vocab_size)` maps each position's hidden state to logits over the vocabulary
- **Logits**: Unnormalized log-probabilities — higher values mean the model thinks that token is more likely
- Output shape: `(B, T, vocab_size)` — one distribution over the vocabulary for each position in the sequence

## Your Task

Implement the `LMHead` module in `exercise.py`:

1. **`__init__`** - Create a LayerNorm and a Linear projection layer
2. **`forward(x)`** - Apply layer norm, then project to vocabulary size

## Verify

```bash
pytest nano-gpt/02_layers/f_lm_head/test_exercise.py -v
```
