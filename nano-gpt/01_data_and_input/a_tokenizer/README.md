# Character-Level Tokenizer

## Concepts

A tokenizer converts raw text into sequences of integers that a model can process,
and converts model output integers back to readable text.

In this exercise you will build a **character-level** tokenizer:

- **Vocabulary**: The set of all unique characters in the text corpus
- **stoi** (string-to-integer): A dictionary mapping each character to a unique integer index
- **itos** (integer-to-string): The reverse mapping from integer index back to character
- **vocab_size**: The number of unique characters
- Characters must be **sorted** to guarantee deterministic, reproducible ordering
- `encode(s)` converts a string into a list of integer token IDs
- `decode(ids)` converts a list of integer token IDs back into a string

## Your Task

Implement three functions in `exercise.py`:

1. **`build_vocab(text)`** - Extract unique characters, sort them, create `stoi`/`itos` mappings
2. **`encode(s, stoi)`** - Convert a string to a list of integers using the mapping
3. **`decode(token_ids, itos)`** - Convert a list of integers back to a string

## Verify

```bash
pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v

# Test individual functions:
pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestBuildVocab
pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestEncode
pytest nano-gpt/01_data_and_input/a_tokenizer/test_exercise.py -v -k TestDecode
```
