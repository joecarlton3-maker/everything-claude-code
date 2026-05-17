"""Anchored walk-forward validation.

Strategy:
  1. For each candidate param combo, run ONE full backtest on the full dataset.
  2. Slice the trade log by entry_time into K (train, test) folds:
       Fold k:  train = trades with entry_time in [t0, split_k)
                test  = trades with entry_time in [split_k, split_{k+1})
  3. For each fold:
       a. Rank candidates by objective on TRAIN trades
       b. Take the winner; record its OOS objective on TEST trades
  4. Return per-fold winners + overall stats.

Why this is correct (and faster than the naïve K×M backtests):
  - The strategy is stateful but path-deterministic: the trades produced
    given a config are a fixed function of the data. Running once on the
    full series and slicing trades by entry time yields exactly the trades
    that WOULD have occurred in a streaming run.
  - Slicing happens AFTER signal generation, so it's just relabeling existing
    trades — no look-ahead.
  - Param selection still uses only train-window trades for the objective.

Caveat: the BarState (TQI, bands, etc.) at the start of fold k's test
window is "warm" — it has seen all bars up to that point. In live trading
this matches reality. In strict cold-start tests it doesn't. We document
this and recommend "anchored" mode (train grows) for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..backtest.runner import run_backtest
from ..config import SatsConfig
from ..strategy.simulator import SimulationResult
from ..strategy.stats import summarize


@dataclass
class WalkForwardFold:
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    winner_params: dict[str, Any]
    train_objective: float
    test_objective: float
    train_stats: dict
    test_stats: dict
    n_candidates: int


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold]
    summary: pd.DataFrame
    n_folds: int = field(init=False)

    def __post_init__(self):
        self.n_folds = len(self.folds)

    @property
    def mean_test_objective(self) -> float:
        if not self.folds:
            return float("nan")
        return float(np.mean([f.test_objective for f in self.folds]))


def _run_one(df: pd.DataFrame, base_cfg: SatsConfig, override: dict[str, Any]):
    """Run one backtest. Return (override, SimulationResult)."""
    from dataclasses import replace as _rep
    cfg = _rep(base_cfg, **override)
    result = run_backtest(df, cfg)
    return override, result.simulation


def _slice_stats(trades: pd.DataFrame, equity_r: pd.Series,
                 t_start: pd.Timestamp, t_end: pd.Timestamp):
    """Compute stats on the subset of trades entered in [t_start, t_end)."""
    mask = (trades["entry_time"] >= t_start) & (trades["entry_time"] < t_end)
    sub_trades = trades[mask].reset_index(drop=True)
    # Equity curve restricted to the same window (resets to 0 at t_start)
    eq_window = equity_r.loc[t_start:t_end]
    if not eq_window.empty:
        eq_window = eq_window - eq_window.iloc[0]
    return summarize(sub_trades, eq_window)


def anchored_walk_forward(
    df: pd.DataFrame,
    base_cfg: SatsConfig,
    grid: list[dict[str, Any]] | dict[str, list[Any]],
    *,
    n_folds: int = 5,
    objective: Callable | str = "calmar",
    n_jobs: int = -1,
    verbose: int = 0,
) -> WalkForwardResult:
    """Run anchored walk-forward.

    Folds 1..K partition the data into (k+1)/(K+1) equal slices. Fold k's
    train window is [start, k/(K+1) * total], test window is
    [k/(K+1) * total, (k+1)/(K+1) * total]. So with n_folds=5 you get
    train sizes 1/6, 2/6, 3/6, 4/6, 5/6 of total data.
    """
    from .objectives import OBJECTIVES
    from .sweep import expand_grid

    combos = expand_grid(grid) if isinstance(grid, dict) else list(grid)
    obj_fn = OBJECTIVES[objective] if isinstance(objective, str) else objective

    # ── 1. Run each candidate once on full data (parallel) ──
    raw = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(_run_one)(df, base_cfg, c) for c in combos
    )
    # raw is list of (override_dict, SimulationResult)

    # ── 2. Build fold boundaries ──
    t_start = df.index[0]
    t_end = df.index[-1]
    total = (t_end - t_start).total_seconds()
    splits = []
    for k in range(n_folds + 1):
        frac = (k + 1) / (n_folds + 1)
        splits.append(t_start + pd.Timedelta(seconds=total * frac))
    # splits[0] = end of train for fold 0, splits[1] = end of test for fold 0, etc.

    folds: list[WalkForwardFold] = []
    summary_rows: list[dict[str, Any]] = []

    for k in range(n_folds):
        train_start = t_start
        train_end = splits[k]
        test_start = splits[k]
        test_end = splits[k + 1]

        # Score every candidate on this fold's train window
        scored = []
        for override, sim in raw:
            train_stats = _slice_stats(sim.trades, sim.equity_r, train_start, train_end)
            score = obj_fn(train_stats)
            scored.append((score, override, sim, train_stats))

        # Winner = highest train objective
        scored.sort(key=lambda r: r[0], reverse=True)
        best_score, best_override, best_sim, best_train_stats = scored[0]

        # OOS evaluation
        test_stats = _slice_stats(best_sim.trades, best_sim.equity_r, test_start, test_end)
        test_score = obj_fn(test_stats)

        fold = WalkForwardFold(
            fold_idx=k,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            winner_params=best_override,
            train_objective=best_score,
            test_objective=test_score,
            train_stats=best_train_stats.to_dict(),
            test_stats=test_stats.to_dict(),
            n_candidates=len(combos),
        )
        folds.append(fold)

        row = {
            "fold": k,
            "train_end": train_end,
            "test_end": test_end,
            "train_obj": best_score,
            "test_obj": test_score,
            "train_trades": best_train_stats.n_trades,
            "test_trades": test_stats.n_trades,
            "train_wr": best_train_stats.win_rate,
            "test_wr": test_stats.win_rate,
            **{f"p_{k}": v for k, v in best_override.items()},
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return WalkForwardResult(folds=folds, summary=summary)
