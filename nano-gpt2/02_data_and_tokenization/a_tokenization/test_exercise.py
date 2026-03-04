import pytest
from exercise import get_encoder, encode, decode, get_eot_token


@pytest.fixture
def encoder():
    return get_encoder()


class TestTokenization:
    def test_roundtrip(self, encoder):
        """Encoding then decoding should recover the original text."""
        text = "Hello, world! This is a test."
        tokens = encode(text, encoder)
        recovered = decode(tokens, encoder)
        assert recovered == text

    def test_vocab_size(self, encoder):
        """GPT-2 tokenizer has 50257 tokens."""
        assert encoder.n_vocab == 50257

    def test_eot_token(self, encoder):
        """The <|endoftext|> token should be 50256."""
        assert get_eot_token(encoder) == 50256

    def test_encode_returns_list_of_ints(self, encoder):
        tokens = encode("hello", encoder)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_hello_is_single_token(self, encoder):
        """'hello' is a common word and should encode to a single BPE token."""
        tokens = encode("hello", encoder)
        assert len(tokens) == 1

    def test_empty_string(self, encoder):
        """Empty string should encode to empty list."""
        tokens = encode("", encoder)
        assert tokens == []

    def test_decode_empty(self, encoder):
        """Decoding empty list should return empty string."""
        text = decode([], encoder)
        assert text == ""

    def test_multiword_encoding(self, encoder):
        """A sentence should encode to multiple tokens."""
        tokens = encode("The quick brown fox jumps over the lazy dog.", encoder)
        assert len(tokens) > 1
        # Roundtrip check
        assert decode(tokens, encoder) == "The quick brown fox jumps over the lazy dog."
