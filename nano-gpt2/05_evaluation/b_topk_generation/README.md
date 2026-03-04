# Top-k Generation

## Concepts

- **Top-k sampling**: Instead of sampling from the entire vocabulary (which can produce incoherent low-probability tokens), restrict sampling to only the `k` most likely tokens. This improves generation quality while maintaining diversity.
- **torch.topk**: Returns the top-k values and their indices from the probability distribution.
- **torch.multinomial**: Samples from the filtered top-k distribution, so only high-probability tokens can be selected.
- **torch.gather**: Maps the sampled index (position within the top-k list) back to the actual vocabulary token ID.
- **Default k=50**: Matches HuggingFace's default. Smaller `k` = more focused/deterministic; `k=1` is greedy decoding.

## Your Task

Implement `generate_topk()` in `exercise.py`:

1. For each iteration: crop context, forward pass, extract last-position logits, softmax to probabilities
2. Apply top-k: `torch.topk` -> `torch.multinomial` -> `torch.gather` to map back to vocab indices
3. Append to sequence and return

## Verify

```bash
pytest nano-gpt2/05_evaluation/b_topk_generation/test_exercise.py -v
```
