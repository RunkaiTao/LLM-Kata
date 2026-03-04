"""
GPT-2 model configuration using a Python dataclass.

In this exercise you will define a GPTConfig dataclass that centralizes
all hyperparameters for the GPT-2 (124M) model. This replaces passing
individual constructor arguments to every module, as was done in the
simpler nano-gpt kata.

Reference: Karpathy's build-nanogpt train_gpt2.py lines 71-77
"""
# TODO: Import dataclass from the dataclasses module
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# YOUR TASK: Define GPTConfig as a dataclass with GPT-2 (124M) defaults
# ---------------------------------------------------------------------------

# TODO: Decorate the class with @dataclass
# TODO: Define five integer fields with GPT-2 (124M) defaults:
#   - block_size: maximum sequence length (1024)
#   - vocab_size: total token count (50257) — 50k BPE merges + 256 bytes + 1 special token
#   - n_layer: number of transformer layers (12)
#   - n_head: number of attention heads (12)
#   - n_embd: embedding dimension (768)
# (use the standard dataclass field syntax: name: type = default)

@dataclass
class GPTConfig:
    # Step 1: block_size = ...  (maximum sequence length, default 1024)
    # Step 2: vocab_size = ...  (total token count, default 50257)
    # Step 3: n_layer = ...     (number of transformer layers, default 12)
    # Step 4: n_head = ...      (number of attention heads, default 12)
    # Step 5: n_embd = ...      (embedding dimension, default 768)
    pass


# Run tests: pytest nano-gpt2/01_model_architecture/a_gpt_config/test_exercise.py -v
