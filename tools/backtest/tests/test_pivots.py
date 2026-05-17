"""Pivot detection tests — Pine `ta.pivothigh` / `ta.pivotlow` semantics."""
import numpy as np
import pandas as pd

from sats.strategy.pivots import last_pivot, pivot_high, pivot_low


def _series(values):
    return pd.Series(values, dtype=float)


class TestPivotHigh:
    def test_simple_peak(self):
        # Peak at index 3, left=2 right=2 → confirmed at index 5
        s = _series([1, 2, 3, 5, 3, 2, 1])
        out = pivot_high(s, 2, 2)
        assert np.isnan(out.iloc[4])  # not enough right bars yet
        assert out.iloc[5] == 5.0      # confirmed at index 5 (3 + right=2)
        assert np.isnan(out.iloc[6])

    def test_no_pivot_on_flat(self):
        s = _series([1, 2, 3, 3, 3, 2, 1])
        # Strict comparison — 3 == 3 fails, so no pivot
        out = pivot_high(s, 2, 2)
        assert out.isna().all()

    def test_no_pivot_when_right_neighbor_higher(self):
        s = _series([1, 2, 3, 5, 6, 5, 4])
        out = pivot_high(s, 2, 2)
        # Index 3 (value 5) has a higher right neighbor (6), so not a pivot
        assert np.isnan(out.iloc[5])
        # Index 4 (value 6) IS a pivot — confirmed at index 6
        assert out.iloc[6] == 6.0


class TestPivotLow:
    def test_simple_trough(self):
        s = _series([5, 4, 3, 1, 3, 4, 5])
        out = pivot_low(s, 2, 2)
        assert out.iloc[5] == 1.0


class TestLastPivot:
    def test_forward_fill(self):
        s = pd.Series([np.nan, 5.0, np.nan, np.nan, 7.0, np.nan])
        out = last_pivot(s)
        assert pd.isna(out.iloc[0])
        assert out.iloc[1] == 5.0
        assert out.iloc[2] == 5.0
        assert out.iloc[3] == 5.0
        assert out.iloc[4] == 7.0
        assert out.iloc[5] == 7.0
