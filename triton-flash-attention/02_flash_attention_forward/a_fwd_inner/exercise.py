"""
Flash Attention forward inner kernel: _attn_fwd_inner.

This Triton JIT function implements the inner loop of Flash Attention's forward
pass. It processes a single block of queries (Q_block, already in SRAM) against
all key-value blocks in the sequence, accumulating the output and maintaining
online softmax statistics.

The STAGE parameter controls which KV-block range to process:
  STAGE == 1: Process blocks strictly LEFT of the causal diagonal (lo=0, hi=block_index_q*BLOCK_SIZE_Q)
  STAGE == 2: Process the DIAGONAL block where causal masking must be applied
  STAGE == 3: Process ALL blocks — used for non-causal (full) attention (lo=0, hi=SEQ_LEN)

For each KV block the loop:
1. Loads K_block and computes QK scores via tl.dot
2. Applies softmax scaling and optional causal mask
3. Updates running max m_i and sum l_i (online softmax)
4. Loads V_block and accumulates output O_block

Reference: triton-flash-attention/triton/flash_attention.py lines 7-79
"""
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention


# ---------------------------------------------------------------------------
# YOUR TASK: Implement _attn_fwd_inner
# ---------------------------------------------------------------------------
@triton.jit
def _attn_fwd_inner(
    O_block,        # (BLOCK_SIZE_Q, HEAD_DIM) — output accumulator
    l_i,            # (BLOCK_SIZE_Q,) — running normalization sum
    m_i,            # (BLOCK_SIZE_Q,) — running maximum
    Q_block,        # (BLOCK_SIZE_Q, HEAD_DIM) — query block (stays in SRAM)
    K_block_ptr,    # block pointer to K (transposed: HEAD_DIM x SEQ_LEN)
    V_block_ptr,    # block pointer to V (SEQ_LEN x HEAD_DIM)
    block_index_q,  # which Q block we are processing (scalar)
    softmax_scale,  # 1 / sqrt(HEAD_DIM)
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,    # 1 = left-of-diagonal, 2 = diagonal (with mask), 3 = full
    offs_q: tl.constexpr,   # (BLOCK_SIZE_Q,) — absolute query token offsets
    offs_kv: tl.constexpr,  # (BLOCK_SIZE_KV,) — KV token offsets within a block
    SEQ_LEN: tl.constexpr,
):
    """
    Inner loop: iterate over KV blocks and update O_block, l_i, m_i.

    Returns:
        (O_block, l_i, m_i) — updated accumulators.

    Steps (before the loop):
        Set lo and hi based on STAGE:
          STAGE == 1: lo = 0,                        hi = block_index_q * BLOCK_SIZE_Q
          STAGE == 2: lo = block_index_q * BLOCK_SIZE_Q,  hi = (block_index_q + 1) * BLOCK_SIZE_Q
                      lo = tl.multiple_of(lo, BLOCK_SIZE_Q)   (hint to compiler)
          STAGE == 3: lo = 0,                        hi = SEQ_LEN
        Advance K_block_ptr to column lo  (use tl.advance(K_block_ptr, (0, lo)))
        Advance V_block_ptr to row lo     (use tl.advance(V_block_ptr, (lo, 0)))

    For start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)  (compiler hint)

        Step 1:  K_block = tl.load(K_block_ptr)
        Step 2:  QK_block = tl.dot(Q_block, K_block)          (Q @ K^T, shape BLOCK_SIZE_Q x BLOCK_SIZE_KV)

        Step 3 (STAGE == 2 only — causal diagonal block):
                 mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
                 QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
                 m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
                 QK_block -= m_ij[:, None]

        Step 3 (all other stages):
                 m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
                 QK_block = QK_block * softmax_scale - m_ij[:, None]

        Step 4:  P_block = tl.math.exp(QK_block)               (attention probabilities, unnormalized)
        Step 5:  l_ij = tl.sum(P_block, 1)                     (row sums of P)
        Step 6:  alpha = tl.math.exp(m_i - m_ij)               (correction for previous l_i)
        Step 7:  l_i = l_i * alpha + l_ij                      (update running sum)
        Step 8:  V_block = tl.load(V_block_ptr)
        Step 9:  P_block = P_block.to(tl.float16)
        Step 10: O_block = O_block * alpha[:, None]             (rescale old output)
        Step 11: O_block = tl.dot(P_block, V_block, O_block)   (fused: P @ V + O_block)
        Step 12: m_i = m_ij
        Step 13: V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
                 K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i
    """
    # --- Set range based on STAGE ---
    # TODO: if STAGE == 1: lo, hi = ...
    # TODO: elif STAGE == 2: lo, hi = ...; lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    # TODO: else: lo, hi = ...

    # TODO: K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # TODO: V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # --- Loop over KV blocks ---
    # TODO: for start_kv in range(lo, hi, BLOCK_SIZE_KV):
    #     start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
    #
    #     Step 1:  K_block = tl.load(K_block_ptr)
    #     Step 2:  QK_block = tl.dot(Q_block, K_block)
    #
    #     if STAGE == 2:
    #         Step 3 (diagonal): mask = ...; QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
    #                            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
    #                            QK_block -= m_ij[:, None]
    #     else:
    #         Step 3 (other):    m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
    #                            QK_block = QK_block * softmax_scale - m_ij[:, None]
    #
    #     Step 4:  P_block = tl.math.exp(QK_block)
    #     Step 5:  l_ij = tl.sum(P_block, 1)
    #     Step 6:  alpha = tl.math.exp(m_i - m_ij)
    #     Step 7:  l_i = l_i * alpha + l_ij
    #     Step 8:  V_block = tl.load(V_block_ptr)
    #     Step 9:  P_block = P_block.to(tl.float16)
    #     Step 10: O_block = O_block * alpha[:, None]
    #     Step 11: O_block = tl.dot(P_block, V_block, O_block)
    #     Step 12: m_i = m_ij
    #     Step 13: advance V_block_ptr and K_block_ptr

    # TODO: return O_block, l_i, m_i
    pass


# ---------------------------------------------------------------------------
# Thin wrapper so tests can call _attn_fwd_inner through the full _attn_fwd
# kernel (imported from 02b once that exercise is complete).
# For now, tests drive _attn_fwd_inner indirectly via a minimal forward kernel
# defined in the test file.
# ---------------------------------------------------------------------------

# Run tests: pytest triton-flash-attention/02_flash_attention_forward/a_fwd_inner/test_exercise.py -v
