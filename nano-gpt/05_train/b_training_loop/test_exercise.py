import torch
import pytest
from exercise import GPTLanguageModel, train_model

VOCAB_SIZE = 26
BLOCK_SIZE = 8
N_EMBD = 32
N_HEAD = 4
N_LAYER = 2
BATCH_SIZE = 4
MAX_ITERS = 100
EVAL_INTERVAL = 50
EVAL_ITERS = 5
LR = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestTrainModel:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(42)
        model = GPTLanguageModel(VOCAB_SIZE, BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, dropout=0.0).to(device)
        # Create small synthetic data -- a repeating pattern so model can learn
        pattern = torch.arange(VOCAB_SIZE)
        train_data = pattern.repeat(50)  # 1300 tokens
        val_data = pattern.repeat(10)  # 260 tokens
        return model, train_data, val_data

    def test_returns_loss_records(self, setup):
        model, train_data, val_data = setup
        records = train_model(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE,
            MAX_ITERS, LR, EVAL_INTERVAL, EVAL_ITERS, device,
        )
        assert isinstance(records, list)
        assert len(records) > 0

    def test_loss_decreases(self, setup):
        model, train_data, val_data = setup
        records = train_model(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE,
            MAX_ITERS, LR, EVAL_INTERVAL, EVAL_ITERS, device,
        )
        first_loss = records[0]["train"]
        last_loss = records[-1]["train"]
        assert last_loss < first_loss, "Training loss should decrease"

    def test_model_parameters_changed(self, setup):
        model, train_data, val_data = setup
        initial_params = {n: p.clone() for n, p in model.named_parameters()}
        train_model(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE,
            MAX_ITERS, LR, EVAL_INTERVAL, EVAL_ITERS, device,
        )
        changed = any(
            not torch.equal(initial_params[n], p) for n, p in model.named_parameters()
        )
        assert changed

    def test_correct_number_of_eval_records(self, setup):
        model, train_data, val_data = setup
        records = train_model(
            model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE,
            MAX_ITERS, LR, EVAL_INTERVAL, EVAL_ITERS, device,
        )
        # Evals at iter 0, 50, and 99 (max_iters-1) = 3 records
        expected = len(
            [i for i in range(MAX_ITERS) if i % EVAL_INTERVAL == 0 or i == MAX_ITERS - 1]
        )
        assert len(records) == expected
