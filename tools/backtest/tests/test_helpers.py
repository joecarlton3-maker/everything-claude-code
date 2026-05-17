"""Unit tests for atomic math primitives (helpers.py).

These verify the Pine-port semantics that the rest of the pipeline depends on.
"""
import numpy as np
import pandas as pd
import pytest

from sats.indicator.helpers import (
    atr,
    clamp,
    efficiency_ratio,
    map_clamp,
    map_clamp_inv,
    rma,
    rsi,
    safe_div,
    true_range,
    volume_zscore,
)


class TestSafeDiv:
    def test_scalar_normal(self):
        assert safe_div(10.0, 2.0) == 5.0

    def test_scalar_div_zero(self):
        assert safe_div(10.0, 0.0, fallback=-1.0) == -1.0

    def test_scalar_nan(self):
        assert safe_div(np.nan, 2.0, fallback=0.0) == 0.0

    def test_series_div_zero(self):
        n = pd.Series([1.0, 2.0, 3.0])
        d = pd.Series([2.0, 0.0, 1.0])
        out = safe_div(n, d, fallback=-9.0)
        assert list(out) == [0.5, -9.0, 3.0]


class TestClamp:
    def test_scalar(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-3, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_series(self):
        s = pd.Series([-1, 0, 5, 10, 11])
        out = clamp(s, 0, 10)
        assert list(out) == [0, 0, 5, 10, 10]


class TestMapClamp:
    def test_midpoint(self):
        # Map [0, 10] → [0, 100]; input 5 → output 50
        assert map_clamp(5.0, 0.0, 10.0, 0.0, 100.0) == 50.0

    def test_clamps_below(self):
        assert map_clamp(-5.0, 0.0, 10.0, 0.0, 100.0) == 0.0

    def test_clamps_above(self):
        assert map_clamp(15.0, 0.0, 10.0, 0.0, 100.0) == 100.0

    def test_inverted_output_range(self):
        # Pine map_clamp DOES allow inverted output ranges (out_lo > out_hi).
        # At v=in_lo we should get out_lo; at v=in_hi we should get out_hi.
        assert map_clamp(0.0, 0.0, 10.0, 100.0, 0.0) == 100.0
        assert map_clamp(10.0, 0.0, 10.0, 100.0, 0.0) == 0.0

    def test_zero_input_range_returns_out_lo(self):
        assert map_clamp(5.0, 3.0, 3.0, 10.0, 20.0) == 10.0


class TestMapClampInv:
    def test_low_input_high_output(self):
        # map_clamp_inv(v, lo, hi, out_high, out_low): at v=lo → out_high
        assert map_clamp_inv(0.0, 0.0, 10.0, 100.0, 0.0) == 100.0

    def test_high_input_low_output(self):
        assert map_clamp_inv(10.0, 0.0, 10.0, 100.0, 0.0) == 0.0


class TestEfficiencyRatio:
    def test_pure_trend_er_one(self):
        # Monotonic series → ER = 1
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        er = efficiency_ratio(s, 5)
        # Bars 0-4 are NaN due to insufficient lookback; bar 5+ should be 1.0
        assert er.iloc[5] == pytest.approx(1.0)
        assert er.iloc[6] == pytest.approx(1.0)

    def test_zigzag_er_zero(self):
        # Pure zigzag: net change zero, max volatility → ER = 0
        s = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
        er = efficiency_ratio(s, 6)
        assert er.iloc[6] == pytest.approx(0.0, abs=1e-9)

    def test_flat_series_returns_zero(self):
        s = pd.Series([5.0] * 10)
        er = efficiency_ratio(s, 5)
        assert er.iloc[-1] == 0.0


class TestRma:
    def test_seed_is_sma(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rma(s, 3)
        # First 2 NaN, 3rd is SMA of first 3 = 2.0
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == pytest.approx(2.0)

    def test_recursive_step(self):
        # After seed, RMA[i] = RMA[i-1] + (1/n)*(s[i] - RMA[i-1])
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rma(s, 3)
        # seed = 2.0 at idx 2; next = 2.0 + (1/3)*(4-2) = 2.6667
        assert out.iloc[3] == pytest.approx(2.0 + (4.0 - 2.0) / 3.0)
        # then = prev + (1/3)*(5 - prev)
        expected_4 = out.iloc[3] + (5.0 - out.iloc[3]) / 3.0
        assert out.iloc[4] == pytest.approx(expected_4)


class TestAtr:
    def test_shape_and_nans(self, df_small):
        a = atr(df_small["high"], df_small["low"], df_small["close"], 14)
        assert len(a) == len(df_small)
        # Should have NaN for first ~14 bars, then valid values
        assert a.iloc[:13].isna().all()
        assert not a.iloc[20:].isna().any()

    def test_strictly_positive_when_valid(self, df_small):
        a = atr(df_small["high"], df_small["low"], df_small["close"], 14)
        assert (a.dropna() > 0).all()


class TestTrueRange:
    def test_simple_case(self):
        # bar 0: tr = high - low (no prev close)
        # bar 1: tr = max(h-l, |h-pc|, |l-pc|)
        h = pd.Series([10.0, 12.0])
        l = pd.Series([8.0, 9.0])
        c = pd.Series([9.0, 11.0])
        tr = true_range(h, l, c)
        assert tr.iloc[0] == 2.0  # h-l only (prev close is NaN)
        # bar 1: h-l = 3, |h-pc|=|12-9|=3, |l-pc|=|9-9|=0 → max = 3
        assert tr.iloc[1] == 3.0


class TestRsi:
    def test_rsi_bounded(self, df_medium):
        r = rsi(df_medium["close"], 14)
        valid = r.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestVolumeZ:
    def test_zero_mean(self, df_medium):
        z = volume_zscore(df_medium["volume"], 20)
        # Rolling z-score should average near zero on stationary data
        assert abs(z.dropna().mean()) < 0.5
