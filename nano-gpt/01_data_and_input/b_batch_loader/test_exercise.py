import torch
import pytest
from exercise import prepare_data, get_batch

BLOCK_SIZE = 8
BATCH_SIZE = 4
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestPrepareData:
    def test_returns_tensors(self):
        encoded = list(range(100))
        train, val = prepare_data(encoded, 0.9)
        assert isinstance(train, torch.Tensor)
        assert isinstance(val, torch.Tensor)

    def test_dtype_is_long(self):
        encoded = list(range(100))
        train, val = prepare_data(encoded, 0.9)
        assert train.dtype == torch.long
        assert val.dtype == torch.long

    def test_split_sizes(self):
        encoded = list(range(100))
        train, val = prepare_data(encoded, 0.9)
        assert len(train) == 90
        assert len(val) == 10

    def test_split_sizes_80_20(self):
        encoded = list(range(100))
        train, val = prepare_data(encoded, 0.8)
        assert len(train) == 80
        assert len(val) == 20

    def test_data_preserved(self):
        """Original values should be preserved in the split"""
        encoded = list(range(50))
        train, val = prepare_data(encoded, 0.9)
        recombined = torch.cat([train, val])
        assert torch.equal(recombined, torch.tensor(encoded, dtype=torch.long))


class TestGetBatch:
    @pytest.fixture
    def data(self):
        torch.manual_seed(42)
        encoded = list(range(200))
        return prepare_data(encoded, 0.9)

    def test_output_shapes(self, data):
        train_data, val_data = data
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    def test_targets_are_shifted_inputs(self, data):
        """y should be x shifted by one position"""
        torch.manual_seed(123)
        train_data, val_data = data
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        # In our range data (0,1,2,...), y[b,t] should equal x[b,t] + 1
        for b in range(BATCH_SIZE):
            for t in range(BLOCK_SIZE):
                assert y[b, t].item() == x[b, t].item() + 1

    def test_val_split(self, data):
        train_data, val_data = data
        x, y = get_batch(val_data, BLOCK_SIZE, BATCH_SIZE, device)
        assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
        # val_data starts at index 180, so values should be >= 180
        assert x.min().item() >= 180

    def test_dtype_is_long(self, data):
        train_data, val_data = data
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_device(self, data):
        train_data, val_data = data
        x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        assert str(x.device).startswith(device.split(":")[0])
        assert str(y.device).startswith(device.split(":")[0])
