import math
import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _load import load

NemotronHConfig = load("01_config_and_primitives", "a_nemotron_h_config").NemotronHConfig
from exercise import NemotronHForCausalLM

HIDDEN_SIZE = 64
VOCAB_SIZE = 256
NUM_LAYERS = 6
HYBRID_PATTERN = "M-M*-E"
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def config():
    return NemotronHConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        hybrid_override_pattern=HYBRID_PATTERN,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        mamba_num_heads=16,
        mamba_head_dim=8,
        mamba_n_groups=2,
        ssm_state_size=16,
        mamba_d_conv=4,
        mamba_expand=2,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        tie_word_embeddings=False,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    return NemotronHForCausalLM(config)


class TestCausalLMInit:
    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_has_model(self, model):
        NemotronHModel = load("03_decoder_and_model", "b_nemotron_h_model").NemotronHModel
        assert isinstance(model.model, NemotronHModel)

    def test_has_lm_head(self, model):
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert model.lm_head.in_features == HIDDEN_SIZE
        assert model.lm_head.out_features == VOCAB_SIZE

    def test_lm_head_no_bias(self, model):
        assert model.lm_head.bias is None


class TestWeightTying:
    def test_no_tying_by_default(self, config):
        config.tie_word_embeddings = False
        model = NemotronHForCausalLM(config)
        assert model.model.embed_tokens.weight is not model.lm_head.weight

    def test_tying_when_enabled(self, config):
        config.tie_word_embeddings = True
        model = NemotronHForCausalLM(config)
        assert model.model.embed_tokens.weight is model.lm_head.weight


class TestForwardPass:
    def test_logits_shape_no_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, loss = model(idx)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        assert loss is None

    def test_logits_shape_with_targets(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        logits, loss = model(idx, targets)
        assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    def test_loss_is_scalar(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        _, loss = model(idx, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_untrained_loss_near_log_vocab(self, model):
        """Random model loss should be approximately ln(vocab_size)."""
        torch.manual_seed(0)
        idx = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN))
        _, loss = model(idx, targets)
        expected = math.log(VOCAB_SIZE)
        assert abs(loss.item() - expected) < 2.0

    def test_different_seq_lengths(self, model):
        for T in [1, 4, 8]:
            idx = torch.randint(0, VOCAB_SIZE, (1, T))
            logits, _ = model(idx)
            assert logits.shape == (1, T, VOCAB_SIZE)

    def test_gradients_flow(self, model):
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        _, loss = model(idx, targets)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
