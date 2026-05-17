"""Dynamic TP tests — verify scaling, floors, and the re-sort invariant."""
import numpy as np
import pandas as pd
import pytest

from sats.indicator.dynamic_tp import compute_dynamic_tp


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="15min")
    return pd.Series(values, index=idx)


class TestDisabled:
    def test_returns_base_unchanged(self):
        tqi = _series([0.5, 0.5, 0.5])
        vol = _series([1.0, 1.0, 1.0])
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.5, max_scale=2.0,
            floor_r1=0.5, ceil_r3=8.0,
            enabled=False,
        )
        assert (r.scale == 1.0).all()
        assert (r.tp1_r == 1.0).all()
        assert (r.tp3_r == 3.0).all()


class TestScalingBounds:
    def test_min_scale_at_zero_quality_zero_vol(self):
        tqi = _series([0.0] * 5)
        vol = _series([0.0] * 5)  # below the 0.5 input floor → vol_comp = 0
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.5, max_scale=2.0,
            floor_r1=0.5, ceil_r3=8.0,
            enabled=True,
        )
        np.testing.assert_allclose(r.scale.values, 0.5)

    def test_max_scale_at_full_quality_high_vol(self):
        tqi = _series([1.0] * 5)
        vol = _series([3.0] * 5)  # above 2.0 → vol_comp = 1
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.5, max_scale=2.0,
            floor_r1=0.5, ceil_r3=8.0,
            enabled=True,
        )
        np.testing.assert_allclose(r.scale.values, 2.0)


class TestSortInvariant:
    def test_tp1_le_tp2_le_tp3_always(self):
        # Random tqi/vol — sort invariant must hold regardless
        np.random.seed(42)
        tqi = _series(np.random.rand(200))
        vol = _series(np.random.rand(200) * 3)
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.5, max_scale=2.0,
            floor_r1=0.5, ceil_r3=8.0,
            enabled=True,
        )
        assert (r.tp1_r <= r.tp2_r).all()
        assert (r.tp2_r <= r.tp3_r).all()


class TestFloorsAndCeilings:
    def test_floor_respected_at_min_scale(self):
        tqi = _series([0.0] * 3)
        vol = _series([0.0] * 3)
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.1,   # would push tp1 to 0.1 without floor
            max_scale=2.0,
            floor_r1=0.5,    # floor should kick in
            ceil_r3=8.0,
            enabled=True,
        )
        assert (r.tp1_r >= 0.5).all()

    def test_ceiling_respected_at_max_scale(self):
        tqi = _series([1.0] * 3)
        vol = _series([3.0] * 3)
        r = compute_dynamic_tp(
            tqi, vol,
            base_tp1_r=1.0, base_tp2_r=2.0, base_tp3_r=3.0,
            tqi_weight=0.6, vol_weight=0.4,
            min_scale=0.5, max_scale=10.0,  # would push tp3 to 30 without ceil
            floor_r1=0.5, ceil_r3=8.0,
            enabled=True,
        )
        assert (r.tp3_r <= 8.0).all()
