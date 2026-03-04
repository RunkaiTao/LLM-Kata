# GPT Model

## Concepts

- **nn.ModuleDict**: GPT-2 stores its transformer components (`wte`, `wpe`, `h`, `ln_f`) in an `nn.ModuleDict` rather than `nn.Sequential`. This allows named access to sub-modules (e.g., `self.transformer.wte`).
- **Weight tying**: The token embedding matrix (`wte.weight`) is shared with the output projection (`lm_head.weight`). This reduces parameters and enforces symmetry between input and output representations.
- **Scaled initialization**: Layers with `NANOGPT_SCALE_INIT` (the output projections in attention and MLP) get their weights scaled by `(2 * n_layer)^-0.5`. This prevents the residual stream's variance from growing with depth. All other Linear layers use `std=0.02`, and Embedding layers also use `std=0.02`.
- **Forward pass**: Token embeddings + position embeddings -> N transformer blocks -> final LayerNorm -> linear projection to vocabulary logits. Cross-entropy loss is computed when targets are provided.

## Your Task

Implement `GPT` in `exercise.py`:

1. `__init__`: Build the model using `nn.ModuleDict` with `wte`, `wpe`, `h` (ModuleList of Blocks), `ln_f`; create `lm_head`; apply weight tying; call `self.apply(self._init_weights)`
2. `_init_weights`: Initialize Linear layers with `Normal(0, 0.02)` (scaled for `NANOGPT_SCALE_INIT` layers), Embedding layers with `Normal(0, 0.02)`, and biases to zeros
3. `forward`: Compute embeddings, pass through blocks, final norm, project to logits, optionally compute loss

## Verify

```bash
pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v

# Test individual parts:
pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestModuleDict
pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestWeightTying
pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestScaledInit
pytest nano-gpt2/01_model_architecture/e_gpt_model/test_exercise.py -v -k TestForwardPass
```
