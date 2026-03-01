# Assemble the GPT Model

## Concepts

The full GPT model assembles all sub-components from the previous exercises:

- **Embeddings**: Reuses the `Embeddings` module (token + position embeddings)
- **Transformer blocks**: `nn.Sequential` of `n_layer` Block modules
- **LM head**: Reuses the `LMHead` module (final LayerNorm + linear projection to logits)

**Weight initialization** (`_init_weights`):
- Linear layers: weights from `Normal(mean=0.0, std=0.02)`, biases set to zero
- Embedding layers: weights from `Normal(mean=0.0, std=0.02)`
- Applied recursively via `self.apply(self._init_weights)`

**Forward pass** transforms token indices into logits:
1. **Embeddings**: `self.embeddings(idx)` `(B, T, C)`
2. **Transformer blocks**: Pass through all blocks `(B, T, C)`
3. **LM head**: `self.lm_head(x)` `(B, T, vocab_size)`

**Loss computation** (when targets are provided):
- Reshape logits from `(B, T, C)` to `(B*T, C)` and targets from `(B, T)` to `(B*T)`
- Compute `F.cross_entropy(logits, targets)`

## Your Task

Implement in `exercise.py`:

1. **`GPTLanguageModel.__init__`** - Compose Embeddings, Blocks, and LMHead; apply weight initialization
2. **`GPTLanguageModel._init_weights`** - Initialize weights for Linear and Embedding modules
3. **`GPTLanguageModel.forward`** - Forward pass from token indices to logits and cross-entropy loss

All sub-component classes (Embeddings, Head, MultiHeadAttention, FeedForward, Block, LMHead) are provided.

## Verify

```bash
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v

# Test individual parts:
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestModelAssembly
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestInitWeights
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestForwardPass
```
