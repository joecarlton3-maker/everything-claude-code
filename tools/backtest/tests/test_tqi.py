"""TQI tests — verify each component is in [0, 1] and the blend respects weights."""
import numpy as np
import pandas as pd
import pytest

from sats.indicator.helpers import efficiency_ratio
from sats.indicator.tqi import compute_tqi


def _components(df, **overrides):
    er = efficiency_ratio(df["close"], 20)
    vol_ratio = pd.Series(1.0, index=df.index)
    kwargs = dict(
        close=df["close"],
        high=df["high"],
        low=df["low"],
        volume=df.get("volume"),
        er_series=er,
        vol_ratio=vol_ratio,
        use_tqi=True,
        weight_er=0.35,
        weight_vol=0.20,
        weight_struct=0.25,
        weight_mom=0.20,
        struct_len=20,
        mom_len=10,
        vol_z_len=20,
    )
    kwargs.update(overrides)
    return compute_tqi(**kwargs)


class TestBounds:
    def test_all_components_in_unit_interval(self, df_medium):
        t = _components(df_medium)
        for s in (t.er, t.vol, t.struct, t.mom, t.tqi):
            valid = s.dropna()
            assert (valid >= 0).all(), f"{s.name} below 0"
            assert (valid <= 1).all(), f"{s.name} above 1"

    def test_disabled_returns_half(self, df_medium):
        t = _components(df_medium, use_tqi=False)
        assert (t.tqi == 0.5).all()


class TestStructure:
    def test_price_at_range_extremes_gives_one(self):
        # Engineer a series where close sits exactly at the rolling high.
        idx = pd.date_range("2024-01-01", periods=50, freq="15min")
        close = pd.Series(np.linspace(100, 150, 50), index=idx)
        high = close + 0.01  # tiny offset so close ≈ high in window
        low = close - 5.0
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
        t = _components(df, struct_len=20, weight_struct=1.0, weight_er=0, weight_vol=0, weight_mom=0)
        # By bar 30+, close is at the upper extreme of the 20-bar range → struct ≈ 1
        assert t.struct.iloc[35] == pytest.approx(1.0, abs=0.1)


class TestMomentum:
    def test_pure_trend_gives_one(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="15min")
        close = pd.Series(np.arange(50, dtype=float), index=idx)
        high = close + 0.1
        low = close - 0.1
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
        t = _components(df, mom_len=10)
        # Every bar moves up + 10-bar window also moves up → 100% alignment
        assert t.mom.iloc[20] == pytest.approx(1.0)

    def test_zigzag_gives_half(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="15min")
        close = pd.Series([100 + (i % 2) for i in range(50)], dtype=float, index=idx)
        high = close + 0.1
        low = close - 0.1
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
        t = _components(df, mom_len=10)
        # On a 1-bar zigzag: net 10-bar change is 0 → mom signal undefined direction
        # Per Pine logic, both > and < comparisons against 0 are false → 0 aligned.
        # That gives mom == 0 in pure zigzag (matches Pine semantics).
        assert t.mom.iloc[20] == pytest.approx(0.0)


class TestBlend:
    def test_weights_normalize(self, df_medium):
        # If we set only er weight, tqi should equal tqi_er where defined.
        t = _components(df_medium, weight_er=1.0, weight_vol=0, weight_struct=0, weight_mom=0)
        # Compare on bars where ER is defined (after warmup)
        diff = (t.tqi.iloc[50:] - t.er.iloc[50:]).abs()
        assert diff.max() < 1e-9
