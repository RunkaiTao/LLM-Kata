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

    Steps:
    1. Call tiktoken.get_encoding("gpt2")
    2. Return the encoder object

    Returns:
        A tiktoken Encoding object for GPT-2.
    """
    # TODO: Implement get_encoder
    return tiktoken.get_encoding("gpt2")


def encode(text: str, encoder) -> list:
    """
    Encode a string into a list of BPE token IDs.

    Args:
        text: The input string to tokenize.
        encoder: A tiktoken Encoding object (from get_encoder()).

    Returns:
        A list of integer token IDs.

    Steps:
    1. Call encoder.encode(text) to tokenize the text
    2. Return the list of token IDs
    """
    # TODO: Implement encode
    return encoder.encode(text)


def decode(tokens: list, encoder) -> str:
    """
    Decode a list of token IDs back into a string.

    Args:
        tokens: A list of integer token IDs.
        encoder: A tiktoken Encoding object.

    Returns:
        The decoded string.

    Steps:
    1. Call encoder.decode(tokens)
    2. Return the decoded string
    """
    # TODO: Implement decode
    return encoder.decode(tokens)


def get_eot_token(encoder) -> int:
    """
    Return the <|endoftext|> special token ID.

    This token is used to separate documents during training.
    For GPT-2, it is token 50256.

    Args:
        encoder: A tiktoken Encoding object.

    Returns:
        The integer ID of the <|endoftext|> token.

    Steps:
    1. Access encoder._special_tokens['<|endoftext|>']
    2. Return it
    """
    # TODO: Implement get_eot_token
    return encoder._special_tokens["<|endoftext|>"]
