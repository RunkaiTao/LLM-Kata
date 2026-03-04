"""
Autoregressive generation with the GPT model.
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Import completed exercise: GPTLanguageModel from 03
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

_GPTBase = load("03_combine_layers", "a_assemble_model").GPTLanguageModel


# ---------------------------------------------------------------------------
# YOUR TASK: Implement generate (extends GPTLanguageModel with generation)
# ---------------------------------------------------------------------------
class GPTLanguageModel(_GPTBase):

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting context of shape (B, T), tensor of token indices.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Tensor of shape (B, T + max_new_tokens) with the generated sequence.

        Steps:
        1. For each of max_new_tokens iterations:
           a. Crop idx to at most the last block_size tokens as idx_cond
              (the model's context window limit)
           b. Get logits from the model's forward pass on idx_cond (ignore loss)
           c. Extract logits for only the last time step -> (B, vocab_size)
           d. Convert to probabilities probs using softmax along the last dimension
              (use F.softmax) -> (B, vocab_size)
           e. Sample idx_next from the probability distribution
              (use torch.multinomial with num_samples=1) -> (B, 1)
           f. Append idx_next to idx along the sequence dimension
              (use torch.cat) -> (B, T+1)
        2. Return idx
        """
        # TODO: Implement generate following the steps above
        # Step 1: for _ in range(max_new_tokens):
        #     a. idx_cond = ...  (idx[:, -self.block_size:])
        #     b. logits, _ = ... (self(idx_cond))
        #     c. logits = ...    (logits[:, -1, :])
        #     d. probs = ...     (F.softmax(logits, dim=-1))
        #     e. idx_next = ...  (torch.multinomial(probs, num_samples=1))
        #     f. idx = ...       (torch.cat((idx, idx_next), dim=1))
        # Step 2: return idx
        pass

# Run tests: pytest nano-gpt/05_inference/a_generate/test_exercise.py -v
