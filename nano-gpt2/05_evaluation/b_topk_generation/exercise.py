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
    1. Crop idx to the last block_size tokens to stay within context window
    2. Forward pass on the cropped context, discard loss
    3. Take logits at the last time step only -> (B, vocab_size)
    4. Convert to probabilities (use F.softmax)
    5. Select the top_k highest-probability tokens and their indices (use torch.topk)
    6. Sample from the top-k distribution (use torch.multinomial)
    7. Map the sampled index back to the original vocab ID (use torch.gather)
    8. Append the new token to the sequence (use torch.cat)

    Return the full sequence
    """
    # TODO: Implement generate_topk following the steps above
    # for _ in range(max_new_tokens):
    #     Step 1: idx_cond = ...              (crop idx to last block_size tokens)
    #     Step 2: logits, _ = ...             (forward pass, discard loss)
    #     Step 3: logits = ...                (last time step only -> (B, vocab_size))
    #     Step 4: probs = ...                 (F.softmax over last dim)
    #     Step 5: topk_probs, topk_indices = ... (torch.topk)
    #     Step 6: ix = ...                    (torch.multinomial from topk_probs)
    #     Step 7: xcol = ...                  (torch.gather to map back to vocab ID)
    #     Step 8: idx = ...                   (torch.cat along dim=1)
    # return idx
    pass

# Run tests: pytest nano-gpt2/05_evaluation/b_topk_generation/test_exercise.py -v
