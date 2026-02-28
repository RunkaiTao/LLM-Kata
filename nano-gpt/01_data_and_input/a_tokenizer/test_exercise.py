import pytest
from exercise import build_vocab, encode, decode


class TestBuildVocab:
    def test_simple_text(self):
        """build_vocab on 'hello' produces 4 unique chars: e, h, l, o"""
        stoi, itos, vocab_size = build_vocab("hello")
        assert vocab_size == 4
        assert set(stoi.keys()) == {"h", "e", "l", "o"}
        assert set(itos.values()) == {"h", "e", "l", "o"}

    def test_sorted_order(self):
        """Characters must be sorted: 'cab' -> a=0, b=1, c=2"""
        stoi, itos, vocab_size = build_vocab("cab")
        assert stoi["a"] == 0
        assert stoi["b"] == 1
        assert stoi["c"] == 2

    def test_itos_is_inverse_of_stoi(self):
        """itos and stoi must be inverses of each other"""
        stoi, itos, vocab_size = build_vocab("the quick brown fox")
        for ch, idx in stoi.items():
            assert itos[idx] == ch

    def test_single_char(self):
        """Edge case: single character repeated"""
        stoi, itos, vocab_size = build_vocab("aaa")
        assert vocab_size == 1
        assert stoi == {"a": 0}

    def test_empty_string(self):
        """Edge case: empty string produces empty vocab"""
        stoi, itos, vocab_size = build_vocab("")
        assert vocab_size == 0
        assert stoi == {}
        assert itos == {}


class TestEncode:
    def test_basic_encode(self):
        """Encode 'abc' with stoi={'a':0,'b':1,'c':2} -> [0,1,2]"""
        stoi = {"a": 0, "b": 1, "c": 2}
        assert encode("abc", stoi) == [0, 1, 2]

    def test_repeated_chars(self):
        """Encode 'aba' -> [0,1,0]"""
        stoi = {"a": 0, "b": 1}
        assert encode("aba", stoi) == [0, 1, 0]

    def test_empty_string(self):
        """Encode empty string -> []"""
        assert encode("", {}) == []

    def test_roundtrip_with_build_vocab(self):
        """Encoding with a vocab built from the same text works"""
        text = "hello world"
        stoi, itos, vs = build_vocab(text)
        encoded = encode(text, stoi)
        assert len(encoded) == len(text)
        assert all(isinstance(i, int) for i in encoded)


class TestDecode:
    def test_basic_decode(self):
        """Decode [0,1,2] with itos={0:'a',1:'b',2:'c'} -> 'abc'"""
        itos = {0: "a", 1: "b", 2: "c"}
        assert decode([0, 1, 2], itos) == "abc"

    def test_empty_list(self):
        """Decode [] -> ''"""
        assert decode([], {}) == ""

    def test_roundtrip(self):
        """encode then decode should recover original string"""
        text = "the quick brown fox"
        stoi, itos, vs = build_vocab(text)
        assert decode(encode(text, stoi), itos) == text
