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
        1. Assert there is at least one shard
        2. Store shards, B, and T as instance attributes
        3. Call self.reset() to initialize shard and position state
        """
        # TODO: Implement __init__ following the steps above
        # Step 1: assert len(shards) > 0
        # Step 2: self.shards = ..., self.B = ..., self.T = ...
        # Step 3: self.reset()
        pass

    def reset(self):
        """
        Reset the loader to the beginning of the first shard.

        Steps:
        1. Set current_shard index to 0
        2. Load tokens from the first shard
        3. Reset current_position to the start
        """
        # TODO: Implement reset following the steps above
        # Step 1: self.current_shard = ...
        # Step 2: self.tokens = ...           (load from self.shards)
        # Step 3: self.current_position = ...
        pass

    def next_batch(self):
        """
        Return the next sequential batch of (inputs, targets).

        Returns:
            A tuple (x, y) where:
            - x has shape (B, T) — input token IDs
            - y has shape (B, T) — target token IDs (shifted by 1)

        Steps:
        1. Extract a contiguous buffer of B*T + 1 tokens starting at current_position
           — the extra +1 token provides the target for the last input position
        2. Split the buffer into inputs x (all but last) and targets y (all but first),
           then reshape both to (B, T) (use .view)
        3. Advance current_position by B*T
        4. If the next batch would overflow the current shard, wrap around to the
           next shard (cycling with modulo) and reset position to 0
        5. Return (x, y)
        """
        # TODO: Implement next_batch following the steps above
        # Step 1: B, T = ...
        #         buf = ...  (slice B*T + 1 tokens from current_position)
        # Step 2: x = ...    (buf[:-1] reshaped to (B, T))
        #         y = ...    (buf[1:] reshaped to (B, T))
        # Step 3: self.current_position += B * T
        # Step 4: if next batch would overflow:
        #             self.current_shard = ...  (cycle with modulo)
        #             self.tokens = ...         (load new shard)
        #             self.current_position = 0
        # return x, y
        pass

# Run tests: pytest nano-gpt2/02_data_and_tokenization/b_data_loader_lite/test_exercise.py -v
