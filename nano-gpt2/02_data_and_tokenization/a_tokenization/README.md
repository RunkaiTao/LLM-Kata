# BPE Tokenization

## Concepts

- **Byte Pair Encoding (BPE)**: Unlike nano-gpt's character-level tokenizer (one token per character), GPT-2 uses BPE — a subword tokenization algorithm that learns common byte pairs from training data. This gives a vocabulary of 50,257 tokens.
- **tiktoken**: OpenAI's fast BPE tokenizer library. `tiktoken.get_encoding("gpt2")` returns the GPT-2 tokenizer.
- **Vocabulary composition**: 256 byte-level tokens (raw bytes) + 50,000 BPE merge tokens (learned subword units) + 1 special `<|endoftext|>` token = 50,257 total.
- **End-of-text token**: `<|endoftext|>` (ID 50256) separates documents during training. The model learns to treat it as a boundary between unrelated text.

## Your Task

Implement four functions in `exercise.py`:

1. `get_encoder()` — return the GPT-2 BPE encoder from tiktoken
2. `encode(text, encoder)` — convert a string into a list of token IDs
3. `decode(tokens, encoder)` — convert token IDs back into a string
4. `get_eot_token(encoder)` — return the `<|endoftext|>` token ID

## Verify

```bash
pytest nano-gpt2/02_data_and_tokenization/a_tokenization/test_exercise.py -v
```
