# Assemble the GPT Model

## Concepts

The full GPT model assembles all sub-components together:

- **Token embedding table**: `nn.Embedding(vocab_size, n_embd)`
- **Position embedding table**: `nn.Embedding(block_size, n_embd)`
- **Transformer blocks**: `nn.Sequential` of `n_layer` Block modules
- **Final LayerNorm**: `nn.LayerNorm(n_embd)` after all blocks
- **Language model head**: `nn.Linear(n_embd, vocab_size)` projects to vocabulary logits

**Weight initialization** (`_init_weights`):
- Linear layers: weights from `Normal(mean=0.0, std=0.02)`, biases set to zero
- Embedding layers: weights from `Normal(mean=0.0, std=0.02)`
- Applied recursively via `self.apply(self._init_weights)`

## Your Task

Implement in `exercise.py`:

1. **`GPTLanguageModel.__init__`** - Create all model components and apply weight initialization
2. **`GPTLanguageModel._init_weights`** - Initialize weights for Linear and Embedding modules

All sub-component classes (Head, MultiHeadAttention, FeedForward, Block) are provided.

## Verify

```bash
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v

# Test individual parts:
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestModelAssembly
pytest nano-gpt/03_combine_layers/a_assemble_model/test_exercise.py -v -k TestInitWeights
```
