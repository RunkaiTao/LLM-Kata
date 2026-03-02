import dataclasses
import pytest
from exercise import GPTConfig


class TestGPTConfig:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(GPTConfig)

    def test_default_block_size(self):
        config = GPTConfig()
        assert config.block_size == 1024

    def test_default_vocab_size(self):
        config = GPTConfig()
        assert config.vocab_size == 50257

    def test_default_n_layer(self):
        config = GPTConfig()
        assert config.n_layer == 12

    def test_default_n_head(self):
        config = GPTConfig()
        assert config.n_head == 12

    def test_default_n_embd(self):
        config = GPTConfig()
        assert config.n_embd == 768

    def test_custom_values(self):
        config = GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=4, n_embd=64)
        assert config.block_size == 32
        assert config.vocab_size == 256
        assert config.n_layer == 2
        assert config.n_head == 4
        assert config.n_embd == 64

    def test_head_size_divisibility(self):
        """n_embd must be evenly divisible by n_head for multi-head attention."""
        config = GPTConfig()
        assert config.n_embd % config.n_head == 0
