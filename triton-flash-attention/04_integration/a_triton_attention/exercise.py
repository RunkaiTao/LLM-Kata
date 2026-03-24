"""
Full Flash Attention 2 integration: TritonAttention.

This exercise wires together all the kernels from previous sections into a
single `torch.autograd.Function` that can be used as a drop-in replacement
for standard attention in any PyTorch model.

By implementing `torch.autograd.Function`:
- `forward()` calls `_attn_fwd` and saves tensors needed for backward
- `backward()` calls `_attn_bwd_preprocess`, `_attn_bwd_dk_dv`, `_attn_bwd_dq`
- Gradients flow automatically through `.apply()`

This is how the real Flash Attention library integrates with PyTorch's autograd.

Reference: triton-flash-attention/triton/flash_attention.py lines 516-661
"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

_mod_fwd = load("02_flash_attention_forward", "b_fwd_kernel")
_attn_fwd = _mod_fwd._attn_fwd

_mod_pre = load("03_flash_attention_backward", "a_bwd_preprocess")
_attn_bwd_preprocess = _mod_pre._attn_bwd_preprocess

_mod_dkdv = load("03_flash_attention_backward", "b_bwd_dkdv")
_attn_bwd_dk_dv = _mod_dkdv._attn_bwd_dk_dv

_mod_dq = load("03_flash_attention_backward", "c_bwd_dq")
_attn_bwd_dq = _mod_dq._attn_bwd_dq

reference_attention = load(
    "01_attention_fundamentals", "c_reference_attention"
).reference_attention


# ---------------------------------------------------------------------------
# YOUR TASK: Implement TritonAttention
# ---------------------------------------------------------------------------
class TritonAttention(torch.autograd.Function):
    """
    Flash Attention 2 as a torch.autograd.Function.

    Usage:
        out = TritonAttention.apply(Q, K, V, causal, softmax_scale)
    """

    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        """
        Run Flash Attention forward pass and save state for backward.

        Args:
            ctx:           autograd context (use ctx.save_for_backward, ctx.<attr> = ...)
            Q, K, V:       float16 CUDA tensors of shape (B, H, T, D).
            causal:        bool — apply causal masking.
            softmax_scale: float — scale factor (typically 1/sqrt(D)).

        Returns:
            O: output tensor of shape (B, H, T, D).

        Steps:
        1.  HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
            HEAD_DIM_V = V.shape[-1]
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
            assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V
        2.  O = torch.empty_like(Q)
        3.  stage = 3 if causal else 1
        4.  grid = lambda args: (
                triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
                BATCH_SIZE * NUM_HEADS,
                1)
        5.  M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN),
                             device=Q.device, dtype=torch.float32)
        6.  Call _attn_fwd[grid](...) with all strides and shape arguments.
            (Use Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3) etc.)
        7.  ctx.save_for_backward(Q, K, V, O, M)
        8.  ctx.grid = grid
            ctx.softmax_scale = softmax_scale
            ctx.HEAD_DIM = HEAD_DIM_K
            ctx.causal = causal
        9.  return O
        """
        # TODO: Step 1: unpack shapes, assert HEAD_DIM consistency
        # TODO: Step 2: O = torch.empty_like(Q)
        # TODO: Step 3: stage = 3 if causal else 1
        # TODO: Step 4: grid = lambda args: (triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), BATCH_SIZE * NUM_HEADS, 1)
        # TODO: Step 5: M = torch.empty((BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32)
        # TODO: Step 6: call _attn_fwd[grid](Q=Q, K=K, V=V, softmax_scale=softmax_scale, M=M, O=O,
        #               stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1), ..., STAGE=stage)
        # TODO: Step 7: ctx.save_for_backward(Q, K, V, O, M)
        # TODO: Step 8: ctx.grid=grid; ctx.softmax_scale=softmax_scale; ctx.HEAD_DIM=HEAD_DIM_K; ctx.causal=causal
        # TODO: Step 9: return O
        pass

    @staticmethod
    def backward(ctx, dO):
        """
        Run Flash Attention backward pass.

        Args:
            ctx: autograd context with saved tensors.
            dO:  gradient of output, shape (B, H, T, D).

        Returns:
            dQ, dK, dV, None, None
            (None for causal and softmax_scale — they have no gradient)

        Steps:
        1.  Q, K, V, O, M = ctx.saved_tensors
        2.  assert dO.is_contiguous()
            assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        3.  dQ = torch.empty_like(Q)
            dK = torch.empty_like(K)
            dV = torch.empty_like(V)
        4.  BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        5.  NUM_WARPS, NUM_STAGES = 4, 3
            BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128
        6.  preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
            D = torch.empty_like(M)
        7.  Call _attn_bwd_preprocess[preprocess_grid](O=O, dO=dO, D=D,
                SEQ_LEN=SEQ_LEN, BLOCK_SIZE_Q=BLOCK_SIZE_MACRO, HEAD_DIM=ctx.HEAD_DIM)
        8.  grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS)
            stage = 3 if ctx.causal else 1
        9.  Call _attn_bwd_dk_dv[grid](Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
                dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
                stride_batch=Q.stride(0), ...,
                NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
                BLOCK_Q=BLOCK_SIZE_MICRO, BLOCK_KV=BLOCK_SIZE_MACRO,
                HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
                num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        10. Call _attn_bwd_dq[grid](Q=Q, K=K, V=V, softmax_scale=ctx.softmax_scale,
                dO=dO, dQ=dQ, dK=dK, dV=dV, M=M, D=D,
                stride_batch=Q.stride(0), ...,
                NUM_HEADS=NUM_HEADS, SEQ_LEN=SEQ_LEN,
                BLOCK_Q=BLOCK_SIZE_MACRO, BLOCK_KV=BLOCK_SIZE_MICRO,
                HEAD_DIM=ctx.HEAD_DIM, STAGE=stage,
                num_warps=NUM_WARPS, num_stages=NUM_STAGES)
        11. return dQ, dK, dV, None, None
        """
        # TODO: Step 1:  Q, K, V, O, M = ctx.saved_tensors
        # TODO: Step 2:  assert dO.is_contiguous()
        #                assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        # TODO: Step 3:  dQ, dK, dV = torch.empty_like(Q/K/V)
        # TODO: Step 4:  BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        # TODO: Step 5:  NUM_WARPS=4; NUM_STAGES=3; BLOCK_SIZE_MICRO=32; BLOCK_SIZE_MACRO=128
        # TODO: Step 6:  preprocess_grid = ...; D = torch.empty_like(M)
        # TODO: Step 7:  call _attn_bwd_preprocess[preprocess_grid](...)
        # TODO: Step 8:  grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS); stage = ...
        # TODO: Step 9:  call _attn_bwd_dk_dv[grid](...)
        # TODO: Step 10: call _attn_bwd_dq[grid](...)
        # TODO: Step 11: return dQ, dK, dV, None, None
        pass


# Run tests: pytest triton-flash-attention/04_integration/a_triton_attention/test_exercise.py -v
