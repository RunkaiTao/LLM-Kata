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
    1. Shift logits and tokens for next-token alignment:
       — remove the last position from logits and the first from tokens
       so that logits[t] predicts tokens[t+1] (use [..., :-1, :] and [..., 1:])
    2. Flatten the shifted logits and tokens, then compute per-token
       cross-entropy loss with no reduction (use F.cross_entropy, reduction='none')
    3. Reshape losses back to (num_candidates, seq_len - 1)
    4. Shift the mask the same way (drop first position) so it aligns
       with the loss positions
    5. Zero out losses in the context region by multiplying with the shifted mask
    6. Compute the average loss per candidate: sum masked losses and divide
       by the number of completion tokens per candidate
    7. Return the index of the candidate with the lowest average loss
       (use .argmin().item())
    """
    # TODO: Implement get_most_likely_row following the steps above
    # Step 1: shift_logits = ...        (logits[..., :-1, :], remove last position)
    #         shift_tokens = ...        (tokens[..., 1:], remove first position)
    # Step 2: flat_shift_logits = ...   (flatten to (-1, vocab_size))
    #         flat_shift_tokens = ...   (flatten to (-1,))
    #         shift_losses = ...        (F.cross_entropy with reduction="none")
    # Step 3: shift_losses = ...        (reshape to (num_candidates, seq_len - 1))
    # Step 4: shift_mask = ...          (mask[..., 1:] to align with losses)
    # Step 5: masked_shift_losses = ... (shift_losses * shift_mask)
    # Step 6: sum_loss = ...            (sum along dim=1)
    #         avg_loss = ...            (sum_loss / shift_mask.sum(dim=1))
    # Step 7: return avg_loss.argmin().item()
    pass

# Run tests: pytest nano-gpt2/05_evaluation/c_hellaswag/test_exercise.py -v
