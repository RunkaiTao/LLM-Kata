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
# TODO: Define the following fields with type int and default values:
#   - block_size: int = 1024   # maximum sequence length
#   - vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 byte tokens + 1 <|endoftext|>
#   - n_layer: int = 12        # number of transformer layers
#   - n_head: int = 12         # number of attention heads
#   - n_embd: int = 768        # embedding dimension

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
