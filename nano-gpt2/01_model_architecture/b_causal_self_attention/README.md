# Causal Self-Attention

## Concepts

- **Batched QKV projection**: Unlike nano-gpt (which uses separate `Head` modules via `nn.ModuleList`), GPT-2 computes Q, K, V for all heads in a single linear layer (`c_attn`) projecting to `3 * n_embd`. This is more efficient.
- **Head reshaping**: After the batched projection, split into Q, K, V and reshape to `(B, n_head, T, head_size)` for parallel head computation.
- **Scaled dot-product attention**: `att = (Q @ K^T) / sqrt(head_size)`, followed by causal masking, softmax, and multiplication with V.
- **Causal mask**: A lower-triangular matrix registered as a buffer (`register_buffer`) that prevents tokens from attending to future positions. Masked positions are filled with `-inf` before softmax.
- **Output projection**: `c_proj` projects the concatenated heads back to `n_embd`. Flagged with `NANOGPT_SCALE_INIT` for scaled initialization in the full model.

## Your Task

Implement `CausalSelfAttention` in `exercise.py`:

1. `__init__`: Create `c_attn` (batched QKV), `c_proj` (output projection with scale init flag), and register the causal mask buffer
2. `forward`: Project to QKV, reshape for multi-head, compute scaled dot-product attention with causal masking, reassemble heads, output projection

## Verify

```bash
pytest nano-gpt2/01_model_architecture/b_causal_self_attention/test_exercise.py -v
```
