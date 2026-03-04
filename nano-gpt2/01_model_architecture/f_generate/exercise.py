"""
Basic autoregressive text generation from the GPT model.

In this exercise you implement the generate() method that produces new tokens
one at a time by feeding the model's own output back as input. This is the
simplest form of generation (multinomial sampling from the full vocabulary).

Top-k sampling and temperature are covered in a later exercise (05/b).

Reference: Karpathy's build-nanogpt train_gpt2.py lines 446-480 (simplified)
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

_GPTBase = load("01_model_architecture", "e_gpt_model").GPT
GPTConfig = load("01_model_architecture", "a_gpt_config").GPTConfig


# ---------------------------------------------------------------------------
# YOUR TASK: Implement generate (extends GPT with generation capability)
# ---------------------------------------------------------------------------
class GPT(_GPTBase):

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting context of shape (B, T), tensor of token indices.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            Tensor of shape (B, T + max_new_tokens) with the generated sequence.

        Steps:
        For each of max_new_tokens iterations:
        1. Crop idx to the last block_size tokens to stay within context window
        2. Run a forward pass on the cropped context, discard the loss
        3. Take logits at the last time step only -> (B, vocab_size)
        4. Convert to probabilities (use F.softmax)
        5. Sample a single next token from the distribution (use torch.multinomial)
        6. Append the sampled token to the sequence (use torch.cat along dim 1)
        Return the full sequence
        """
        # TODO: Implement autoregressive generation following the steps above
        # for _ in range(max_new_tokens):
        #     Step 1: idx_cond = ...  (crop idx to last block_size tokens)
        #     Step 2: logits, _ = ... (forward pass, discard loss)
        #     Step 3: logits = ...    (take last time step only -> (B, vocab_size))
        #     Step 4: probs = ...     (F.softmax over last dim)
        #     Step 5: idx_next = ...  (torch.multinomial, num_samples=1)
        #     Step 6: idx = ...       (torch.cat along dim=1)
        # return idx
        pass

# Run tests: pytest nano-gpt2/01_model_architecture/f_generate/test_exercise.py -v
