"""
BPE tokenization using tiktoken for GPT-2.

Unlike the simpler nano-gpt kata which uses character-level tokenization
(one token per character), GPT-2 uses Byte Pair Encoding (BPE) with a
vocabulary of 50,257 tokens. This exercise wraps the tiktoken library
to provide the tokenization functions needed for training.

The GPT-2 tokenizer includes:
- 256 byte-level tokens (raw bytes)
- 50,000 BPE merge tokens (learned subword units)
- 1 special <|endoftext|> token (document separator)

Reference: Karpathy's build-nanogpt train_gpt2.py lines 205-206, 322
"""
import tiktoken


# ---------------------------------------------------------------------------
# YOUR TASK: Implement the tokenization wrapper functions
# ---------------------------------------------------------------------------

def get_encoder():
    """
    Return the GPT-2 BPE encoder from tiktoken.

    Returns:
        A tiktoken Encoding object for GPT-2.

    Hint: Use tiktoken.get_encoding with the "gpt2" encoding name.
    """
    # TODO: Implement get_encoder
    # return ...  (tiktoken.get_encoding with "gpt2")
    pass


def encode(text: str, encoder) -> list:
    """
    Encode a string into a list of BPE token IDs.

    Args:
        text: The input string to tokenize.
        encoder: A tiktoken Encoding object (from get_encoder()).

    Returns:
        A list of integer token IDs.

    Hint: The encoder object has an encode method that takes a string.
    """
    # TODO: Implement encode
    # return ...  (use encoder.encode)
    pass


def decode(tokens: list, encoder) -> str:
    """
    Decode a list of token IDs back into a string.

    Args:
        tokens: A list of integer token IDs.
        encoder: A tiktoken Encoding object.

    Returns:
        The decoded string.

    Hint: The encoder object has a decode method that takes a list of token IDs.
    """
    # TODO: Implement decode
    # return ...  (use encoder.decode)
    pass


def get_eot_token(encoder) -> int:
    """
    Return the <|endoftext|> special token ID.

    This token is used to separate documents during training.
    For GPT-2, it is token 50256.

    Args:
        encoder: A tiktoken Encoding object.

    Returns:
        The integer ID of the <|endoftext|> token.

    Hint: The encoder stores special tokens in a _special_tokens dict,
    keyed by the token string (e.g. '<|endoftext|>').
    """
    # TODO: Implement get_eot_token
    # return ...  (access encoder._special_tokens dict with "<|endoftext|>" key)
    pass

# Run tests: pytest nano-gpt2/02_data_and_tokenization/a_tokenization/test_exercise.py -v
