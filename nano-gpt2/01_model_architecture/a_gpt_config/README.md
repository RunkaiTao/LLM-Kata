# GPT Config

## Concepts

- **Dataclass**: A Python `@dataclass` decorator auto-generates `__init__`, `__repr__`, and other boilerplate from field definitions. This is the standard way to define configuration objects.
- **GPT-2 (124M) defaults**: The smallest GPT-2 model uses `block_size=1024`, `vocab_size=50257`, `n_layer=12`, `n_head=12`, `n_embd=768`.
- **Centralized config**: Instead of passing individual hyperparameters to every module (as in the simpler nano-gpt kata), a single `GPTConfig` object is shared across all components.
- **vocab_size = 50,257**: 50,000 BPE merges + 256 byte-level tokens + 1 special `<|endoftext|>` token.

## Your Task

Implement `GPTConfig` in `exercise.py`:

1. Import `dataclass` from the `dataclasses` module
2. Decorate the class with `@dataclass`
3. Define five fields with type `int` and their default values: `block_size`, `vocab_size`, `n_layer`, `n_head`, `n_embd`

## Verify

```bash
pytest nano-gpt2/01_model_architecture/a_gpt_config/test_exercise.py -v
```
