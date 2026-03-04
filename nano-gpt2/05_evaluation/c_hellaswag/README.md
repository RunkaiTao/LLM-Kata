# HellaSwag Evaluation

## Concepts

- **HellaSwag benchmark**: A common-sense reasoning benchmark. The model is given a context and 4 possible completions, and must select the most likely one. This tests the model's language understanding without any fine-tuning.
- **Per-token loss scoring**: For each candidate, compute the cross-entropy loss at every token position. The completion with the lowest average loss is the model's prediction.
- **Logit shifting**: Logits at position `t` predict the token at position `t+1`. Both logits and tokens must be shifted to align predictions with targets: `shift_logits = logits[..., :-1, :]`, `shift_tokens = tokens[..., 1:]`.
- **Mask-based scoring**: A binary mask marks which tokens belong to the completion region (1) vs. the shared context (0). Only completion tokens are scored — the context is shared across all candidates.

## Your Task

Implement `get_most_likely_row()` in `exercise.py`:

1. Shift logits and tokens for next-token alignment
2. Compute per-token cross-entropy loss with `reduction='none'`
3. Apply the mask to score only completion tokens
4. Average the masked loss per candidate and return the index of the lowest

## Verify

```bash
pytest nano-gpt2/05_evaluation/c_hellaswag/test_exercise.py -v
```
