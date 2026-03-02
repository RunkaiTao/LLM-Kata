"""
Sequential batch data loader for GPT-2 training.

Unlike the simpler nano-gpt kata which randomly samples batches, GPT-2 training
uses sequential loading through large tokenized datasets stored as "shards".
The DataLoaderLite processes data sequentially, advancing a position pointer,
and rotates through multiple shards when one is exhausted.

In the real build-nanogpt, shards are numpy files on disk. For testability,
this exercise uses a list of torch tensors to simulate shards.

Key concepts:
- Sequential (not random) batch extraction
- Targets are inputs shifted by 1 (next-token prediction)
- Shard rotation when current shard is exhausted
- Buffer extraction: buf[:-1] for inputs, buf[1:] for targets

Reference: Karpathy's build-nanogpt train_gpt2.py lines 214-252
"""
import torch


# ---------------------------------------------------------------------------
# YOUR TASK: Implement DataLoaderLite
# ---------------------------------------------------------------------------
class DataLoaderLite:

    def __init__(self, shards: list, B: int, T: int):
        """
        Args:
            shards: A list of 1-D torch.long tensors, each representing a shard
                    of pre-tokenized data.
            B: Batch size (number of sequences per batch).
            T: Sequence length (context window).

        Steps:
        1. Assert len(shards) > 0 — must have at least one shard
        2. Store self.shards = shards
        3. Store self.B = B and self.T = T
        4. Call self.reset()
        """
        # TODO: Implement __init__ following the steps above
        assert len(shards) > 0, "must have at least one shard"
        self.shards = shards
        self.B = B
        self.T = T
        self.reset()

    def reset(self):
        """
        Reset the loader to the beginning of the first shard.

        Steps:
        1. self.current_shard = 0
        2. self.tokens = self.shards[self.current_shard]
        3. self.current_position = 0
        """
        # TODO: Implement reset following the steps above
        self.current_shard = 0
        self.tokens = self.shards[self.current_shard]
        self.current_position = 0

    def next_batch(self):
        """
        Return the next sequential batch of (inputs, targets).

        Returns:
            A tuple (x, y) where:
            - x has shape (B, T) — input token IDs
            - y has shape (B, T) — target token IDs (shifted by 1)

        Steps:
        1. B, T = self.B, self.T
        2. Extract a buffer: buf = self.tokens[self.current_position : self.current_position + B*T + 1]
           — we need B*T + 1 tokens to create B*T inputs and B*T targets (shifted by 1)
        3. x = buf[:-1].view(B, T)  — inputs
        4. y = buf[1:].view(B, T)   — targets (next token for each position)
        5. Advance position: self.current_position += B * T
        6. If the next batch would exceed the current shard length:
             self.current_shard = (self.current_shard + 1) % len(self.shards)
             self.tokens = self.shards[self.current_shard]
             self.current_position = 0
        7. Return (x, y)
        """
        # TODO: Implement next_batch following the steps above
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.shards[self.current_shard]
            self.current_position = 0
        return x, y
