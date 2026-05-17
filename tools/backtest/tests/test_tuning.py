"""Tests for objectives, sweep, and walk-forward."""
import math

import pandas as pd
import pytest

from sats.config import SatsConfig
from sats.tuning import anchored_walk_forward, expand_grid, grid_search
from sats.tuning.objectives import calmar, sharpe_like, total_r


class TestExpandGrid:
    def test_two_keys(self):
        g = {"a": [1, 2], "b": ["x", "y"]}
        out = expand_grid(g)
        assert len(out) == 4
        assert {"a": 1, "b": "x"} in out
        assert {"a": 2, "b": "y"} in out

    def test_single_key(self):
        out = expand_grid({"a": [1, 2, 3]})
        assert out == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_empty_grid(self):
        out = expand_grid({})
        # itertools.product() of nothing yields one empty tuple
        assert out == [{}]


class TestObjectives:
    def test_total_r_zero_trades_returns_neg_inf(self):
        assert total_r({"n_trades": 0, "total_r": 0}) == -math.inf

    def test_sharpe_zero_trades_returns_neg_inf(self):
        assert sharpe_like({"n_trades": 0, "sharpe_like": 1.5}) == -math.inf

    def test_calmar_normal(self):
        s = {"n_trades": 50, "total_r": 10.0, "max_dd_r": -2.0}
        assert calmar(s) == pytest.approx(5.0)

    def test_calmar_dd_floor_prevents_inflation(self):
        # Tiny DD shouldn't yield massive Calmar
        s = {"n_trades": 50, "total_r": 10.0, "max_dd_r": -0.01}
        # With dd_floor=0.5, calmar = 10 / 0.5 = 20
        assert calmar(s) == pytest.approx(20.0)

    def test_calmar_low_trade_count_returns_neg_inf(self):
        s = {"n_trades": 5, "total_r": 10.0, "max_dd_r": -1.0}
        assert calmar(s, min_trades=20) == -math.inf


class TestGridSearch:
    def test_small_sweep_returns_ranked(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom")
        grid = {"quality_strength": [0.0, 0.5], "base_mult": [1.5, 2.5]}
        results = grid_search(df_medium, cfg, grid, objective="total_r", n_jobs=1)
        assert len(results) == 4
        # Sorted descending by total_r
        assert results["total_r"].iloc[0] >= results["total_r"].iloc[-1]
        # Has expected columns
        for col in ("quality_strength", "base_mult", "n_trades", "win_rate"):
            assert col in results.columns

    def test_preset_custom_means_overrides_apply(self, df_medium):
        """Regression: with preset='Custom', sweep params must produce different results.
        With a non-Custom preset, params like base_mult get silently overridden.
        """
        cfg = SatsConfig(timeframe_minutes=15, preset="Custom")
        grid = {"base_mult": [1.0, 3.0], "sl_atr_mult": [0.5, 3.0]}
        results = grid_search(df_medium, cfg, grid, objective="total_r", n_jobs=1)
        # 4 combos should produce at least 2 distinct trade counts
        # (if preset overrides params, they'd all be identical)
        assert results["n_trades"].nunique() >= 2, \
            "params not affecting results — preset likely overriding them"


class TestWalkForward:
    def test_anchored_smoke(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15)
        # Tiny grid + 2 folds → fast
        grid = {"quality_strength": [0.0, 0.5]}
        result = anchored_walk_forward(
            df_medium, cfg, grid,
            n_folds=2, objective="total_r", n_jobs=1,
        )
        assert len(result.folds) == 2
        for f in result.folds:
            assert "quality_strength" in f.winner_params
            assert f.train_end <= f.test_start
            assert f.test_start < f.test_end

    def test_fold_windows_anchor_to_start(self, df_medium):
        cfg = SatsConfig(timeframe_minutes=15)
        result = anchored_walk_forward(
            df_medium, cfg, {"quality_strength": [0.4]},
            n_folds=3, objective="total_r", n_jobs=1,
        )
        # All train windows start at the same point (anchored)
        start = result.folds[0].train_start
        for f in result.folds:
            assert f.train_start == start
