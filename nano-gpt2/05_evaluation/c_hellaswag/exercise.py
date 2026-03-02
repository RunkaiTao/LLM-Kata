"""
HellaSwag evaluation: selecting the most likely completion.

HellaSwag is a common-sense reasoning benchmark where the model is given
a context and 4 possible completions. The model selects the completion
with the lowest average cross-entropy loss (i.e., the one it finds most likely).

This exercise implements get_most_likely_row(), which:
1. Shifts logits and tokens to align predictions with targets
2. Computes per-token cross-entropy loss
3. Masks to only score the completion region (not the shared context)
4. Averages the loss per completion and returns the one with lowest loss

Reference: Karpathy's build-nanogpt train_gpt2.py lines 258-275
"""
import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# YOUR TASK: Implement get_most_likely_row
# ---------------------------------------------------------------------------
def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """
    Given tokens, mask, and logits for multiple completion candidates,
    return the index of the completion with the lowest average loss.

    Args:
        tokens: Token IDs of shape (num_candidates, seq_len).
                Each row is context + one completion candidate.
        mask: Binary mask of shape (num_candidates, seq_len).
              1 in the completion region, 0 in the context region.
        logits: Model output logits of shape (num_candidates, seq_len, vocab_size).

    Returns:
        The index (int) of the most likely completion (0-indexed).

    Steps:
    1. Shift logits to align with next-token prediction:
       shift_logits = logits[..., :-1, :].contiguous()
       — logits at position t predict the token at position t+1
    2. Shift tokens similarly:
       shift_tokens = tokens[..., 1:].contiguous()
    3. Flatten for cross_entropy:
       flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
       flat_shift_tokens = shift_tokens.view(-1)
    4. Compute per-token loss (no reduction):
       shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    5. Reshape losses back to (num_candidates, seq_len - 1):
       shift_losses = shift_losses.view(tokens.size(0), -1)
    6. Shift the mask to match:
       shift_mask = mask[..., 1:].contiguous()
       — the mask must be shifted because we're scoring at positions 1..T-1
    7. Apply mask (zero out context positions):
       masked_shift_losses = shift_losses * shift_mask
    8. Compute average loss per completion:
       sum_loss = masked_shift_losses.sum(dim=1)
       avg_loss = sum_loss / shift_mask.sum(dim=1)
    9. Return the index of the completion with the lowest average loss:
       return avg_loss.argmin().item()
    """
    # TODO: Implement get_most_likely_row following the steps above
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()
