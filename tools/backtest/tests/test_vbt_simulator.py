"""Tests for the vectorbt single-target simulator.

Focus: SimulationResult shape matches the numba simulator, exit reasons
classify correctly, R-multiples are in the right ballpark.

NOT a strict parity test against the three-target simulator (different exit
logic by design). We DO assert that both engines produce sensible numbers
on the same data.
"""
import pytest
import pandas as pd
import numpy as np

from sats.backtest.runner import run_backtest
from sats.config import SatsConfig


class TestSingleTargetExitsAreShape:
    def test_runs_without_error(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        result = run_backtest(df_medium, cfg)
        # Should return same shape as three-target
        assert hasattr(result.simulation, "trades")
        assert hasattr(result.simulation, "equity_r")
        assert hasattr(result.simulation, "in_trade")

    def test_trades_have_expected_columns(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        result = run_backtest(df_medium, cfg)
        if len(result.simulation.trades) == 0:
            pytest.skip("no trades on this synthetic slice")
        cols = result.simulation.trades.columns
        for required in ("entry_time", "exit_time", "direction", "entry", "sl",
                         "tp1", "realized_r", "exit_reason"):
            assert required in cols

    def test_tp2_tp3_are_nan_in_single_target(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        result = run_backtest(df_medium, cfg)
        if len(result.simulation.trades) == 0:
            pytest.skip("no trades")
        assert result.simulation.trades["tp2"].isna().all()
        assert result.simulation.trades["tp3"].isna().all()


class TestExitReasonClassification:
    def test_realized_r_is_finite(self, df_medium):
        """vectorbt's from_signals uses close-only stop checking. Bars that gap
        through the SL can exit with realized_r < -1. That's a known limitation
        documented in vbt_simulator.py — see module docstring.

        Test just asserts R values are finite and not absurd (no nan/inf, no
        wildly impossible numbers from a bug)."""
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        result = run_backtest(df_medium, cfg)
        if len(result.simulation.trades) == 0:
            pytest.skip("no trades")
        r = result.simulation.trades["realized_r"]
        assert r.notna().all(), "got NaN R values"
        assert np.isfinite(r).all(), "got inf R values"
        assert r.min() >= -10, f"absurd loss suggests bug: {r.min()}"
        assert r.max() <= 10, f"absurd gain suggests bug: {r.max()}"

    def test_tp1_exits_at_or_above_tp1_r(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        result = run_backtest(df_medium, cfg)
        if len(result.simulation.trades) == 0:
            pytest.skip("no trades")
        tp1_trades = result.simulation.trades[result.simulation.trades["exit_reason"] == "tp1"]
        if len(tp1_trades) > 0:
            # TP1 hits: should be >= tp1_r (allow some upside slippage)
            tp1_r = tp1_trades["tp1_r"]
            realized = tp1_trades["realized_r"]
            assert (realized >= tp1_r - 0.05).all(), "TP1 exit fell short of tp1_r"


class TestEngineDispatch:
    def test_three_target_vs_single_target_differ(self, df_medium):
        """Both engines should run on the same data and produce DIFFERENT results
        (since they use different exit logic). If they're identical the dispatch is broken.
        """
        cfg3 = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="three_target")
        cfg1 = SatsConfig(timeframe_minutes=15, preset="Custom", exit_mode="single_target")
        r3 = run_backtest(df_medium, cfg3)
        r1 = run_backtest(df_medium, cfg1)
        # The trades likely differ in count AND in realized R distribution.
        # If they're identical the dispatch is broken.
        same_count = r3.stats.n_trades == r1.stats.n_trades
        same_total_r = abs(r3.stats.total_r - r1.stats.total_r) < 1e-6
        assert not (same_count and same_total_r), \
            "single_target and three_target produced identical results — dispatch broken"
