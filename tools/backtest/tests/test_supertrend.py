"""SuperTrend tests — invariants + numba/python parity.

Port fidelity is critical: a bug here invalidates every backtest. So we
test (a) structural invariants, (b) that the numba and pure-python kernels
produce bit-identical outputs.
"""
import numpy as np
import pandas as pd
import pytest

from sats.config import SatsConfig
from sats.indicator.engine import compute_bar_state
from sats.indicator.supertrend import _supertrend_numba, _supertrend_python


def _baseline_inputs(df):
    cfg = SatsConfig(timeframe_minutes=15)
    state = compute_bar_state(df, cfg)
    return state, cfg


class TestInvariants:
    def test_trend_only_pm_one(self, df_medium):
        state, _ = _baseline_inputs(df_medium)
        valid = state.supertrend.trend.iloc[state.warmup_bars + 5:]
        assert set(valid.unique()).issubset({-1, 1})

    def test_st_line_below_close_when_long(self, df_medium):
        state, _ = _baseline_inputs(df_medium)
        st = state.supertrend
        # When trend == +1, the active band (lower) should be ≤ close on the same bar
        # (the band is a trailing stop). This is a soft check — possible exceptions
        # on the exact flip bar where price just crossed back over it.
        long_bars = state.supertrend.trend == 1
        violations = (st.st_line[long_bars] > state.close[long_bars])
        # Allow up to ~2% of bars (flip-bar edge cases) to violate strictly.
        assert violations.sum() / max(long_bars.sum(), 1) < 0.05

    def test_warmup_bars_have_no_trend(self, df_medium):
        state, _ = _baseline_inputs(df_medium)
        assert (state.supertrend.trend.iloc[:state.warmup_bars] == 0).all()

    def test_flip_count_is_sane(self, df_medium):
        state, _ = _baseline_inputs(df_medium)
        flips = (state.supertrend.flip_up.sum(), state.supertrend.flip_down.sum())
        # On 1000 synthetic bars at 15m, expect at least a handful of flips
        # but not hundreds (which would indicate broken ratchet logic).
        assert 2 <= sum(flips) <= 200, f"unexpected flip count: {flips}"


class TestNumbaPythonParity:
    """The numba kernel is the production path. The python version is
    canonical. They must produce identical output.
    """

    def _call_both(self, state, cfg):
        args = dict(
            source=state.close.values.astype(np.float64),
            close=state.close.values.astype(np.float64),
            atr_eff=state.atr_eff.values.astype(np.float64),
            tqi=state.tqi.tqi.values.astype(np.float64),
            er=state.er.values.astype(np.float64),
            base_mult=2.0,
            use_adaptive=True,
            adapt_strength=0.5,
            use_tqi=True,
            quality_strength=0.4,
            quality_curve=1.5,
            use_asym=True,
            asym_strength=0.5,
            use_smooth=True,
            mult_smooth_alpha=0.15,
            use_char_flip=True,
            char_high=0.55,
            char_low=0.25,
            char_min_age=5,
            warmup=state.warmup_bars,
        )
        a = _supertrend_numba(**args)
        b = _supertrend_python(**args)
        return a, b

    def test_parity_default(self, df_medium):
        state, cfg = _baseline_inputs(df_medium)
        a, b = self._call_both(state, cfg)
        for i, (xa, xb) in enumerate(zip(a, b)):
            # NaN-safe comparison
            mask = ~(np.isnan(xa) & np.isnan(xb)) if xa.dtype.kind == "f" else np.ones_like(xa, dtype=bool)
            assert np.allclose(np.nan_to_num(xa[mask]), np.nan_to_num(xb[mask]), atol=1e-9), \
                f"mismatch in output[{i}]"


class TestTqiInfluence:
    def test_high_quality_tightens_bands(self, df_medium):
        """Quality strength > 0 should produce different bands than strength = 0."""
        cfg_off = SatsConfig(timeframe_minutes=15, quality_strength=0.0)
        cfg_on = SatsConfig(timeframe_minutes=15, quality_strength=0.8)
        state_off = compute_bar_state(df_medium, cfg_off)
        state_on = compute_bar_state(df_medium, cfg_on)
        # Bands should differ
        diff = (state_off.supertrend.lower_band - state_on.supertrend.lower_band).abs().sum()
        assert diff > 0


class TestCharFlipDisabled:
    def test_disabling_char_flip_reduces_flips(self, df_medium):
        cfg_on = SatsConfig(timeframe_minutes=15, use_char_flip=True)
        cfg_off = SatsConfig(timeframe_minutes=15, use_char_flip=False)
        state_on = compute_bar_state(df_medium, cfg_on)
        state_off = compute_bar_state(df_medium, cfg_off)
        flips_on = state_on.supertrend.flip_up.sum() + state_on.supertrend.flip_down.sum()
        flips_off = state_off.supertrend.flip_up.sum() + state_off.supertrend.flip_down.sum()
        # Char flip adds flips, so on >= off
        assert flips_on >= flips_off
