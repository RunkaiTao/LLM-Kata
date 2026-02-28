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


class TestModelAssembly:
    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        return GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER)

    def test_has_token_embedding(self, model):
        assert hasattr(model, "token_embedding_table")
        assert isinstance(model.token_embedding_table, torch.nn.Embedding)
        assert model.token_embedding_table.num_embeddings == VOCAB_SIZE
        assert model.token_embedding_table.embedding_dim == N_EMBD

    def test_has_position_embedding(self, model):
        assert hasattr(model, "position_embedding_table")
        assert isinstance(model.position_embedding_table, torch.nn.Embedding)
        assert model.position_embedding_table.num_embeddings == BLOCK_SIZE
        assert model.position_embedding_table.embedding_dim == N_EMBD

    def test_has_blocks(self, model):
        assert hasattr(model, "blocks")
        assert isinstance(model.blocks, torch.nn.Sequential)
        assert len(model.blocks) == N_LAYER

    def test_has_final_layer_norm(self, model):
        assert hasattr(model, "ln_f")
        assert isinstance(model.ln_f, torch.nn.LayerNorm)

    def test_has_lm_head(self, model):
        assert hasattr(model, "lm_head")
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert model.lm_head.in_features == N_EMBD
        assert model.lm_head.out_features == VOCAB_SIZE

    def test_forward_produces_correct_shape(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, loss = model(idx)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_forward_with_targets_produces_loss(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, loss = model(idx, targets)
        assert loss is not None
        assert loss.dim() == 0  # scalar

    def test_forward_without_targets_no_loss(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, loss = model(idx)
        assert loss is None


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
        assert model.token_embedding_table.weight.std().item() < 0.1
        assert model.position_embedding_table.weight.std().item() < 0.1
