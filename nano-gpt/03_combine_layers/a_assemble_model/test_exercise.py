import math
import torch
import pytest
from exercise import GPTLanguageModel

VOCAB_SIZE = 26
BLOCK_SIZE = 16
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 2
SEQ_LEN = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestInitWeights:
    def test_linear_weights_are_small(self):
        torch.manual_seed(42)
        model = GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER)
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() == 2:
                assert param.std().item() < 0.1, f"{name} std too large"

    def test_linear_biases_are_zero(self):
        torch.manual_seed(42)
        model = GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER)
        for name, param in model.named_parameters():
            if "bias" in name and "ln" not in name:
                assert torch.allclose(param, torch.zeros_like(param)), f"{name} bias not zero"

    def test_embedding_weights_are_small(self):
        torch.manual_seed(42)
        model = GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER)
        assert model.embeddings.token_embedding_table.weight.std().item() < 0.1
        assert model.embeddings.position_embedding_table.weight.std().item() < 0.1


class TestForwardPass:
    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, dropout=0.0).to(device)

    def test_logits_shape_no_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        logits, loss = model(idx)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        assert loss is None

    def test_logits_shape_with_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        logits, loss = model(idx, targets)
        assert logits.shape == (BATCH_SIZE * SEQ_LEN, VOCAB_SIZE)

    def test_loss_is_scalar(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        _, loss = model(idx, targets)
        assert loss.dim() == 0

    def test_loss_is_positive(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=device)
        _, loss = model(idx, targets)
        assert loss.item() > 0

    def test_untrained_loss_near_neg_log_vocab(self, model):
        """
        For a randomly initialized model, loss should be approximately
        -ln(1/vocab_size) = ln(vocab_size) ~ 3.26 for vocab_size=26
        """
        torch.manual_seed(0)
        idx = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN), device=device)
        _, loss = model(idx, targets)
        expected = math.log(VOCAB_SIZE)
        assert abs(loss.item() - expected) < 1.0  # generous tolerance

    def test_loss_none_without_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (1, 4), device=device)
        _, loss = model(idx)
        assert loss is None

    def test_different_sequence_lengths(self, model):
        """Forward pass should work for any T <= block_size"""
        for T in [1, 4, BLOCK_SIZE]:
            idx = torch.randint(0, VOCAB_SIZE, (1, T), device=device)
            logits, _ = model(idx)
            assert logits.shape[-1] == VOCAB_SIZE

    def test_gradients_flow_through_loss(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (2, 4), device=device)
        targets = torch.randint(0, VOCAB_SIZE, (2, 4), device=device)
        _, loss = model(idx, targets)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad
