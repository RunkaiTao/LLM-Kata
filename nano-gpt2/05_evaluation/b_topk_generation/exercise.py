"""
Top-k sampling for higher quality text generation.

Basic multinomial sampling (01/f) samples from the entire vocabulary,
which can produce low-probability (incoherent) tokens. Top-k sampling
restricts sampling to only the k most likely tokens, improving quality.

The process for each generated token:
1. Forward pass to get logits
2. Extract last position logits
3. Convert to probabilities with softmax
4. Take top-k probabilities and their indices
5. Sample from the top-k distribution
6. Map back to the original vocabulary index using gather

Reference: Karpathy's build-nanogpt train_gpt2.py lines 457-475
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Import completed exercises: GPT, GPTConfig
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

GPT = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement generate_topk
# ---------------------------------------------------------------------------
def generate_topk(model, idx: torch.Tensor, max_new_tokens: int, top_k: int = 50) -> torch.Tensor:
    """
    Generate tokens using top-k sampling.

    Args:
        model: A GPT model instance.
        idx: Starting context of shape (B, T), tensor of token indices.
        max_new_tokens: Number of new tokens to generate.
        top_k: Number of top tokens to sample from (default 50, matching HuggingFace).

    Returns:
        Tensor of shape (B, T + max_new_tokens) with the generated sequence.

    Steps:
    For each of max_new_tokens iterations:
    1. Crop idx to at most the last model.config.block_size tokens as idx_cond
    2. Forward pass: logits, _ = model(idx_cond)
       — logits is (B, T', vocab_size)
    3. Extract last position: logits = logits[:, -1, :]  -> (B, vocab_size)
    4. Convert to probabilities: probs = F.softmax(logits, dim=-1)  -> (B, vocab_size)
    5. Get top-k: topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
       -> each (B, top_k)
    6. Sample from top-k: ix = torch.multinomial(topk_probs, 1)
       -> (B, 1)  — index into the top-k list
    7. Map back to vocab: xcol = torch.gather(topk_indices, -1, ix)
       -> (B, 1)  — the actual token ID
    8. Append: idx = torch.cat((idx, xcol), dim=1)

    Return idx
    """
    # TODO: Implement generate_topk following the steps above
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        idx = torch.cat((idx, xcol), dim=1)
    return idx
