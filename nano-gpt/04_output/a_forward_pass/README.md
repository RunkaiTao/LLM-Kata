# The Forward Pass

## Concepts

The forward pass transforms token indices into logits (unnormalized probabilities):

1. **Token embeddings**: Look up each token's vector `(B, T, C)`
2. **Position embeddings**: Look up position vectors using `torch.arange(T)` `(T, C)`
3. **Sum**: `x = tok_emb + pos_emb` `(B, T, C)`
4. **Transformer blocks**: Pass through all blocks `(B, T, C)`
5. **Final LayerNorm**: `(B, T, C)`
6. **LM head**: Project to vocabulary size `(B, T, vocab_size)`

**Loss computation** (when targets are provided):
- Reshape logits from `(B, T, C)` to `(B*T, C)` and targets from `(B, T)` to `(B*T)`
- Compute `F.cross_entropy(logits, targets)`

**Device handling**: Use `idx.device` for position indices so it works on any device.

## Your Task

Implement `GPTLanguageModel.forward()` in `exercise.py`. The `__init__` and all sub-components are provided.

## Verify

```bash
pytest nano-gpt/04_output/a_forward_pass/test_exercise.py -v
```
