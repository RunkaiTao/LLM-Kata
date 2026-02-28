"""
Character-level tokenizer for nano-GPT.

A tokenizer converts raw text into sequences of integers that a model
can process, and converts model output integers back to readable text.
"""


def build_vocab(text: str) -> tuple[dict, dict, int]:
    """
    Build vocabulary mappings from a text corpus.

    Steps:
    1. Extract all unique characters from the text.
    2. Sort them to create a deterministic ordering.
    3. Create stoi: a dict mapping each character to its integer index.
    4. Create itos: a dict mapping each integer index to its character.
    5. Compute vocab_size as the number of unique characters.

    Args:
        text: The full text corpus as a string.

    Returns:
        A tuple of (stoi, itos, vocab_size).
    """
    unique_chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(unique_chars)}
    itos = {i: ch for i, ch in enumerate(unique_chars)}
    vocab_size = len(unique_chars)
    return stoi, itos, vocab_size


def encode(s: str, stoi: dict) -> list[int]:
    """
    Encode a string into a list of integer token IDs.

    Args:
        s: The string to encode.
        stoi: The string-to-integer mapping dictionary.

    Returns:
        A list of integers representing the encoded string.
    """
    return [stoi[ch] for ch in s]


def decode(token_ids: list[int], itos: dict) -> str:
    """
    Decode a list of integer token IDs back into a string.

    Args:
        token_ids: A list of integers to decode.
        itos: The integer-to-string mapping dictionary.

    Returns:
        The decoded string.
    """
    return ''.join(itos[i] for i in token_ids)
