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
    # TODO: Implement build_vocab following the steps above
    # Step 1: chars = ...       (sorted list of unique characters from text)
    # Step 2: stoi = ...        (dict: char -> index)
    # Step 3: itos = ...        (dict: index -> char)
    # Step 4: vocab_size = ...  (number of unique characters)
    # return (stoi, itos, vocab_size)
    pass


def encode(s: str, stoi: dict) -> list[int]:
    """
    Encode a string into a list of integer token IDs.

    Args:
        s: The string to encode.
        stoi: The string-to-integer mapping dictionary.

    Returns:
        A list of integers representing the encoded string.
    """
    # TODO: Implement encode
    # return ...  (list comprehension: stoi[c] for each character c in s)
    pass


def decode(token_ids: list[int], itos: dict) -> str:
    """
    Decode a list of integer token IDs back into a string.

    Args:
        token_ids: A list of integers to decode.
        itos: The integer-to-string mapping dictionary.

    Returns:
        The decoded string.
    """
    # TODO: Implement decode
    # return ...  (join itos[i] for each token i in token_ids)
    pass

# Run tests: pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v
# Test individual functions:
# pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestBuildVocab
# pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestEncode
# pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestDecode
