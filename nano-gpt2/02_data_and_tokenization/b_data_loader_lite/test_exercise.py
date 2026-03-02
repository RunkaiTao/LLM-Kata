import torch
import pytest
from exercise import DataLoaderLite

B = 2
T = 4


@pytest.fixture
def single_shard():
    """A single shard with sequential token IDs 0..99."""
    return [torch.arange(100, dtype=torch.long)]


@pytest.fixture
def two_shards():
    """Two shards with distinguishable token ranges."""
    shard0 = torch.arange(0, 50, dtype=torch.long)
    shard1 = torch.arange(1000, 1050, dtype=torch.long)
    return [shard0, shard1]


class TestDataLoaderLite:
    def test_batch_shapes(self, single_shard):
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x, y = loader.next_batch()
        assert x.shape == (B, T)
        assert y.shape == (B, T)

    def test_targets_are_shifted_inputs(self, single_shard):
        """y should be x shifted by 1 position (next-token prediction)."""
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x, y = loader.next_batch()
        # For sequential data 0,1,2,...: y should be x + 1
        assert torch.equal(y, x + 1)

    def test_sequential_batches_differ(self, single_shard):
        """Two consecutive next_batch() calls should return different data."""
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x1, _ = loader.next_batch()
        x2, _ = loader.next_batch()
        assert not torch.equal(x1, x2)

    def test_sequential_batches_are_adjacent(self, single_shard):
        """Second batch should start where first batch ended."""
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x1, _ = loader.next_batch()
        x2, _ = loader.next_batch()
        # First batch starts at 0, second batch starts at B*T
        assert x1[0, 0].item() == 0
        assert x2[0, 0].item() == B * T

    def test_shard_rotation(self, two_shards):
        """After exhausting the first shard, loader should move to the second."""
        loader = DataLoaderLite(two_shards, B=2, T=4)
        # First shard has 50 tokens, each batch needs B*T+1 = 9 tokens
        # After enough batches, should rotate to shard 1 (tokens starting at 1000)
        prev_x = None
        found_shard1 = False
        for _ in range(20):
            x, y = loader.next_batch()
            if x[0, 0].item() >= 1000:
                found_shard1 = True
                break
        assert found_shard1, "Loader never rotated to second shard"

    def test_reset(self, single_shard):
        """After reset, loader should return to the beginning."""
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x1, _ = loader.next_batch()
        loader.next_batch()  # advance
        loader.reset()
        x_after_reset, _ = loader.next_batch()
        assert torch.equal(x1, x_after_reset)

    def test_dtype_is_long(self, single_shard):
        loader = DataLoaderLite(single_shard, B=B, T=T)
        x, y = loader.next_batch()
        assert x.dtype == torch.long
        assert y.dtype == torch.long
