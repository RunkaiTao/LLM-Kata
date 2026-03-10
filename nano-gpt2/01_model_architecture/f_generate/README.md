# Generate

## Concepts

- **Autoregressive generation**: The model generates one token at a time, feeding each new token back as input for the next step. This loop continues for `max_new_tokens` iterations.
- **Context cropping**: The input must be cropped to the last `block_size` tokens before each forward pass. Without this, the model would fail on sequences longer than its maximum context window.
- **Multinomial sampling**: After computing softmax probabilities over the vocabulary, `torch.multinomial` randomly samples the next token. This introduces stochasticity — the same prompt can produce different outputs.
- **This is the simplest form**: No temperature, no top-k filtering. Those refinements come in later exercises (05/b).

## Your Task

Implement the `generate` method on the `GPT` class in `exercise.py`:

1. For each iteration: crop context to `block_size`, forward pass, extract last-position logits, softmax to probabilities, sample with `torch.multinomial`, concatenate to sequence

## Verify

```bash
pytest nano-gpt2/01_model_architecture/f_generate/test_exercise.py -v
```
