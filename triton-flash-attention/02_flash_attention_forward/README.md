# 02 — Flash Attention Forward Pass

This section builds up to the Triton Flash Attention forward kernel one concept
at a time. Exercises `a1`–`a5` isolate each Triton building block used in
`_attn_fwd_inner`; then `a` and `b` combine them into the real kernels.

## Exercise Map

| Exercise | Triton concept | What you build | Key API |
|----------|---------------|----------------|---------|
| **a1_matmul_block** | Block matrix multiply | `Q @ K^T` in SRAM | `tl.make_block_ptr`, `tl.load`, `tl.dot`, `tl.store` |
| **a2_scale_causal_mask** | Scaling + conditional mask | `QK * scale + mask` | `tl.arange`, `tl.where`, broadcasting |
| **a3_online_softmax_block** | Online softmax update | `m_ij, alpha, P, l_new` | `tl.max`, `tl.sum`, `tl.math.exp`, `tl.maximum` |
| **a4_output_accumulate** | Fused accumulate with rescale | `O = O*alpha + P@V` | `tl.dot(A, B, acc)`, `.to(tl.float16)` |
| **a5_block_ptr_loop** | Pointer loop pattern | Loop + `tl.advance` | `tl.advance`, `tl.multiple_of`, `range` loop |
| **a_fwd_inner** | Full inner kernel | Combines a1–a5 | All of the above |
| **b_fwd_kernel** | Full forward kernel | Outer kernel + epilogue | `tl.program_id`, autotune |

## How the pieces fit together

```
_attn_fwd_inner combines all 5 building blocks:

┌─ a5: loop over KV blocks with tl.advance ─────────────────────┐
│                                                                 │
│   for each KV block:                                            │
│     ┌─ a1: K_block = tl.load(K_ptr)                            │
│     │      QK = tl.dot(Q_block, K_block)     # Q @ K^T         │
│     │                                                           │
│     ├─ a2: QK_scaled = QK * scale            # + causal mask   │
│     │      mask = offs_q >= offs_kv                             │
│     │      QK_scaled += tl.where(mask, 0, -1e6)                │
│     │                                                           │
│     ├─ a3: m_ij = maximum(m_i, max(QK_scaled))                 │
│     │      alpha = exp(m_i - m_ij)                              │
│     │      P = exp(QK_scaled - m_ij)                            │
│     │      l_i = l_i * alpha + sum(P)                           │
│     │                                                           │
│     └─ a4: O = O * alpha + tl.dot(P, V_block)   # accumulate  │
│                                                                 │
│     K_ptr = tl.advance(K_ptr, (0, BLOCK_KV))   # next block   │
│     V_ptr = tl.advance(V_ptr, (BLOCK_KV, 0))                   │
└─────────────────────────────────────────────────────────────────┘
```

## Running tests

```bash
# Run all tests in this section
pytest triton-flash-attention/02_flash_attention_forward/ -v

# Run exercises in order
pytest triton-flash-attention/02_flash_attention_forward/a1_matmul_block/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/a2_scale_causal_mask/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/a3_online_softmax_block/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/a4_output_accumulate/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/a5_block_ptr_loop/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/a_fwd_inner/test_exercise.py -v
pytest triton-flash-attention/02_flash_attention_forward/b_fwd_kernel/test_exercise.py -v
```
