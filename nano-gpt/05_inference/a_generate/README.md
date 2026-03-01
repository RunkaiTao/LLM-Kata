# Autoregressive Text Generation

## Concepts

Autoregressive generation predicts one token at a time, appending it to the sequence:

- **Context window cropping**: If the sequence exceeds `block_size`, crop to the last
  `block_size` tokens to prevent errors
- **Last time step logits**: Only `logits[:, -1, :]` (shape `(B, vocab_size)`) matter
  for predicting the next token
- **Softmax** converts logits to probabilities
- **`torch.multinomial`** samples from the probability distribution (stochastic generation)
- **`torch.cat`** appends the new token to the running sequence
- The `idx` tensor grows by one column each iteration

## Your Task

Implement `GPTLanguageModel.generate()` in `exercise.py`. The full model with `__init__` and `forward` is provided.

## Verify

```bash
pytest nano-gpt/05_inference/a_generate/test_exercise.py -v
```
