import torch
import pytest
from exercise import get_most_likely_row

VOCAB_SIZE = 50


class TestHellaSwag:
    def test_returns_int(self):
        """Result should be a Python int."""
        tokens = torch.randint(0, VOCAB_SIZE, (4, 10))
        mask = torch.zeros(4, 10, dtype=torch.long)
        mask[:, 5:] = 1  # last 5 positions are completion
        logits = torch.randn(4, 10, VOCAB_SIZE)
        result = get_most_likely_row(tokens, mask, logits)
        assert isinstance(result, int)

    def test_result_in_range(self):
        """Result should be in [0, num_candidates)."""
        tokens = torch.randint(0, VOCAB_SIZE, (4, 10))
        mask = torch.zeros(4, 10, dtype=torch.long)
        mask[:, 5:] = 1
        logits = torch.randn(4, 10, VOCAB_SIZE)
        result = get_most_likely_row(tokens, mask, logits)
        assert 0 <= result < 4

    def test_correct_with_perfect_logits(self):
        """If logits perfectly predict one completion, that row should be selected."""
        seq_len = 8
        num_candidates = 4
        correct_row = 2

        # Create tokens: all rows share context, differ in completion
        tokens = torch.randint(0, VOCAB_SIZE, (num_candidates, seq_len))
        mask = torch.zeros(num_candidates, seq_len, dtype=torch.long)
        mask[:, 4:] = 1  # positions 4-7 are completion

        # Create logits that strongly predict the correct row's tokens
        logits = torch.randn(num_candidates, seq_len, VOCAB_SIZE) * 0.01  # low confidence
        # For the correct row, make logits strongly predict the next token
        for t in range(seq_len - 1):
            if mask[correct_row, t + 1] == 1:  # only in completion region
                logits[correct_row, t, tokens[correct_row, t + 1]] = 10.0

        result = get_most_likely_row(tokens, mask, logits)
        assert result == correct_row

    def test_mask_only_scores_completion(self):
        """Changing context tokens (where mask=0) should not affect the result."""
        seq_len = 10
        tokens = torch.randint(0, VOCAB_SIZE, (4, seq_len))
        mask = torch.zeros(4, seq_len, dtype=torch.long)
        mask[:, 6:] = 1  # only last 4 positions matter

        logits = torch.randn(4, seq_len, VOCAB_SIZE)

        result1 = get_most_likely_row(tokens, mask, logits)

        # Change context tokens (positions 0-5, where mask=0)
        tokens2 = tokens.clone()
        tokens2[:, :5] = torch.randint(0, VOCAB_SIZE, (4, 5))
        result2 = get_most_likely_row(tokens2, mask, logits)

        assert result1 == result2

    def test_four_row_standard_format(self):
        """Standard HellaSwag format: 4 candidates, varying lengths."""
        # Simulate 4 completions of different quality
        seq_len = 12
        tokens = torch.randint(0, VOCAB_SIZE, (4, seq_len))
        mask = torch.zeros(4, seq_len, dtype=torch.long)
        mask[:, 8:] = 1  # 4-token completion

        # Make row 1 the "best" by creating logits that predict it well
        logits = torch.zeros(4, seq_len, VOCAB_SIZE)
        for row in range(4):
            for t in range(seq_len - 1):
                if mask[row, t + 1] == 1:
                    # Give row 1 much better predictions
                    if row == 1:
                        logits[row, t, tokens[row, t + 1]] = 10.0
                    else:
                        logits[row, t, tokens[row, t + 1]] = 1.0

        result = get_most_likely_row(tokens, mask, logits)
        assert result == 1

    def test_two_candidates(self):
        """Should work with any number of candidates, not just 4."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 6))
        mask = torch.zeros(2, 6, dtype=torch.long)
        mask[:, 3:] = 1
        logits = torch.randn(2, 6, VOCAB_SIZE)
        result = get_most_likely_row(tokens, mask, logits)
        assert result in [0, 1]
