"""Objective functions for parameter sweeps.

Each takes a BacktestStats (or its dict form) and returns a float
to MAXIMIZE. Higher = better.

Calmar: trades off raw P&L against max drawdown. Recommended default —
penalizes strategies that grind out gains with one catastrophic loss.

All functions return -inf for degenerate cases (zero trades, etc.) so
the sweep ranks them last instead of crashing.
"""
from __future__ import annotations

import math
from typing import Callable

from ..strategy.stats import BacktestStats


def _as_dict(stats) -> dict:
    return stats.to_dict() if isinstance(stats, BacktestStats) else stats


def total_r(stats) -> float:
    s = _as_dict(stats)
    if s["n_trades"] == 0:
        return -math.inf
    return float(s["total_r"])


def sharpe_like(stats) -> float:
    """Per-trade Sharpe-like: avg R / std R. Not annualized."""
    s = _as_dict(stats)
    if s["n_trades"] < 5:
        return -math.inf
    return float(s["sharpe_like"])


def calmar(stats, *, min_trades: int = 20, dd_floor: float = 0.5) -> float:
    """Total R / |max DD|. With a floor on |DD| so tiny-drawdown lucky runs
    on small samples don't generate absurd values.

    Returns -inf if n_trades < min_trades (avoids picking winners off noise).
    """
    s = _as_dict(stats)
    if s["n_trades"] < min_trades:
        return -math.inf
    dd = abs(s["max_dd_r"])
    if dd < dd_floor:
        dd = dd_floor
    return float(s["total_r"] / dd)


OBJECTIVES: dict[str, Callable] = {
    "total_r": total_r,
    "sharpe_like": sharpe_like,
    "calmar": calmar,
}
