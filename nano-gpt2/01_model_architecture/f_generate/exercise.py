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
        1. For each of max_new_tokens iterations:
           a. Crop idx to at most the last self.config.block_size tokens as idx_cond
              — prevents exceeding the model's maximum context window
           b. Get logits from self(idx_cond), ignore loss -> logits is (B, T', vocab_size)
           c. Extract logits at the last time step only: logits[:, -1, :] -> (B, vocab_size)
           d. Convert to probabilities: probs = F.softmax(logits, dim=-1) -> (B, vocab_size)
           e. Sample next token: idx_next = torch.multinomial(probs, num_samples=1) -> (B, 1)
           f. Append to sequence: idx = torch.cat((idx, idx_next), dim=1) -> (B, T+1)
        2. Return idx
        """
        # TODO: Implement autoregressive generation following the steps above
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
