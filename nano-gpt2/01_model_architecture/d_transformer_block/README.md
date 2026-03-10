# Transformer Block

## Concepts

- **Pre-norm architecture**: GPT-2 applies LayerNorm *before* each sub-layer (attention, MLP), not after. This is known as "pre-norm" and improves training stability compared to the original Transformer's "post-norm".
- **Residual connections**: Each sub-layer's output is added back to its input: `x = x + sublayer(norm(x))`. This allows gradients to flow directly through the network and enables training of deep models.
- **Two sub-layers per block**: (1) LayerNorm + CausalSelfAttention, (2) LayerNorm + MLP. The pattern is `x = x + attn(ln_1(x))` then `x = x + mlp(ln_2(x))`.

## Your Task

Implement `Block` in `exercise.py`:

1. `__init__`: Create `ln_1`, `attn` (CausalSelfAttention), `ln_2`, and `mlp` (MLP). Sub-component classes are provided via imports.
2. `forward`: Apply pre-norm attention with residual, then pre-norm MLP with residual

## Verify

```bash
pytest nano-gpt2/01_model_architecture/d_transformer_block/test_exercise.py -v
```
