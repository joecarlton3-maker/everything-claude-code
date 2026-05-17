"""Grid search over SATS config parameters.

Each row of the output is one parameter combo with its full backtest stats
plus the chosen objective value. Sort by `objective` descending to find
winners — but always look at the top 10-20% as a cohort, not just the best
single combo (single bests are usually noise).

Parallelism via joblib. First call per worker JIT-compiles the numba
kernels (~1-2s); subsequent calls are fast.
"""
from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Any, Callable, Iterable

import pandas as pd
from joblib import Parallel, delayed

from ..backtest.runner import run_backtest
from ..config import SatsConfig


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """{a:[1,2], b:[x,y]} → [{a:1,b:x},{a:1,b:y},{a:2,b:x},{a:2,b:y}]."""
    keys = list(grid.keys())
    out: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        out.append(dict(zip(keys, combo)))
    return out


def _run_one(df: pd.DataFrame, base_cfg: SatsConfig, override: dict[str, Any]) -> dict[str, Any]:
    cfg = replace(base_cfg, **override)
    result = run_backtest(df, cfg)
    s = result.stats.to_dict()
    # Flatten: param overrides + selected stats
    row = dict(override)
    row.update({
        "n_trades": s["n_trades"],
        "win_rate": s["win_rate"],
        "avg_r": s["avg_r"],
        "total_r": s["total_r"],
        "max_dd_r": s["max_dd_r"],
        "profit_factor": s["profit_factor"],
        "sharpe_like": s["sharpe_like"],
        "payoff_ratio": s["payoff_ratio"],
        "bars_per_trade": s["bars_per_trade"],
        "longest_loss_streak": s["longest_loss_streak"],
    })
    return row


def grid_search(
    df: pd.DataFrame,
    base_cfg: SatsConfig,
    grid: dict[str, list[Any]] | Iterable[dict[str, Any]],
    *,
    objective: Callable | str = "calmar",
    n_jobs: int = -1,
    verbose: int = 0,
) -> pd.DataFrame:
    """Run a backtest for every combination in `grid` and return ranked results.

    Args:
        df: OHLCV data.
        base_cfg: defaults for any param not overridden in the grid.
        grid: either a dict {param_name: [values]} or pre-expanded list of dicts.
        objective: function (stats_dict → float) or name from OBJECTIVES.
        n_jobs: joblib parallelism (-1 = all cores).
        verbose: joblib verbosity (0 = silent, 10 = progress bar).

    Returns:
        DataFrame sorted by `objective` descending.
    """
    from .objectives import OBJECTIVES

    combos: list[dict[str, Any]] = (
        expand_grid(grid) if isinstance(grid, dict) else list(grid)
    )

    if isinstance(objective, str):
        obj_fn = OBJECTIVES[objective]
        obj_name = objective
    else:
        obj_fn = objective
        obj_name = getattr(objective, "__name__", "objective")

    rows = Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
        delayed(_run_one)(df, base_cfg, c) for c in combos
    )

    result = pd.DataFrame(rows)
    result[obj_name] = result.apply(obj_fn, axis=1)
    result = result.sort_values(obj_name, ascending=False).reset_index(drop=True)
    return result
