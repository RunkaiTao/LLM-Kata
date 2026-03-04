import math
import pytest
from exercise import get_lr

MAX_LR = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 100
MAX_STEPS = 1000


class TestLRScheduler:
    def test_starts_near_zero(self):
        """Step 0 should give approximately max_lr / warmup_steps (small but not zero)."""
        lr = get_lr(0, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        expected = MAX_LR * 1 / WARMUP_STEPS
        assert abs(lr - expected) < 1e-10

    def test_linear_warmup_midpoint(self):
        """Midway through warmup, lr should be approximately max_lr / 2."""
        it = WARMUP_STEPS // 2
        lr = get_lr(it, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        expected = MAX_LR * (it + 1) / WARMUP_STEPS
        assert abs(lr - expected) < 1e-10

    def test_reaches_max_at_warmup_end(self):
        """At the last warmup step, lr should equal max_lr."""
        lr = get_lr(WARMUP_STEPS - 1, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        expected = MAX_LR * WARMUP_STEPS / WARMUP_STEPS
        assert abs(lr - expected) < 1e-10

    def test_warmup_start_is_max_lr(self):
        """At exactly warmup_steps, lr should still be max_lr (start of cosine)."""
        lr = get_lr(WARMUP_STEPS, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        assert abs(lr - MAX_LR) < 1e-10

    def test_decays_to_min_at_max_steps(self):
        """At max_steps, lr should equal min_lr."""
        lr = get_lr(MAX_STEPS, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        assert abs(lr - MIN_LR) < 1e-10

    def test_cosine_midpoint(self):
        """At the midpoint of cosine decay, lr should be approximately (max_lr + min_lr) / 2."""
        mid = (WARMUP_STEPS + MAX_STEPS) // 2
        lr = get_lr(mid, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        expected_mid = (MAX_LR + MIN_LR) / 2
        assert abs(lr - expected_mid) < 1e-4

    def test_beyond_max_steps(self):
        """After max_steps, lr should remain at min_lr."""
        lr = get_lr(MAX_STEPS + 1000, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        assert lr == MIN_LR

    def test_monotonic_during_warmup(self):
        """LR should monotonically increase during warmup."""
        prev = 0
        for it in range(WARMUP_STEPS):
            lr = get_lr(it, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
            assert lr > prev
            prev = lr

    def test_monotonic_during_decay(self):
        """LR should monotonically decrease during cosine decay."""
        prev = MAX_LR + 1
        for it in range(WARMUP_STEPS, MAX_STEPS + 1):
            lr = get_lr(it, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
            assert lr <= prev
            prev = lr
