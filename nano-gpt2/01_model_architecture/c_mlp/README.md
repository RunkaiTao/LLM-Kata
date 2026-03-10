# MLP (Feed-Forward Network)

## Concepts

- **4x expansion**: The MLP first expands the embedding dimension by 4x (`n_embd -> 4*n_embd`), then projects back (`4*n_embd -> n_embd`). This gives the network more capacity to learn transformations.
- **GELU activation**: GPT-2 uses `GELU(approximate='tanh')`, not ReLU (as in nano-gpt) or exact GELU. The tanh approximation was standard at the time GPT-2 was published.
- **NANOGPT_SCALE_INIT**: The output projection (`c_proj`) is flagged with this attribute. Later, `GPT._init_weights` uses it to scale down initialization, preventing the residual stream variance from growing with depth.

## Your Task

Implement `MLP` in `exercise.py`:

1. `__init__`: Create `c_fc` (expansion), `gelu` activation with tanh approximation, and `c_proj` (projection with scale init flag)
2. `forward`: Apply `c_fc -> gelu -> c_proj`

## Verify

```bash
pytest nano-gpt2/01_model_architecture/c_mlp/test_exercise.py -v
```
