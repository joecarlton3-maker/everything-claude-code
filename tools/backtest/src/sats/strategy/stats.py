"""Summary statistics computed from a TradeLog.

Kept dependency-free (no quantstats, no vectorbt) so this works as the
single source of truth for parameter sweeps. Add fancier reporting later
in `sats.backtest.runner`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestStats:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_r: float
    median_r: float
    total_r: float
    max_dd_r: float
    expectancy: float           # mean R per trade
    payoff_ratio: float         # avg_win / avg_loss
    longest_win_streak: int
    longest_loss_streak: int
    profit_factor: float        # sum(wins) / sum(|losses|)
    sharpe_like: float          # mean / std of per-trade R (not annualized)
    bars_per_trade: float
    pct_exits_by_reason: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity - running_max
    return float(dd.min())


def _longest_streak(series: pd.Series, condition) -> int:
    streak = 0
    longest = 0
    for v in series:
        if condition(v):
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    return longest


def summarize(trades: pd.DataFrame, equity_r: pd.Series) -> BacktestStats:
    n = len(trades)
    if n == 0:
        return BacktestStats(
            n_trades=0, n_wins=0, n_losses=0, win_rate=0.0,
            avg_r=0.0, median_r=0.0, total_r=0.0, max_dd_r=0.0,
            expectancy=0.0, payoff_ratio=0.0,
            longest_win_streak=0, longest_loss_streak=0,
            profit_factor=0.0, sharpe_like=0.0, bars_per_trade=0.0,
            pct_exits_by_reason={},
        )

    r = trades["realized_r"]
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    payoff = abs(avg_win / avg_loss) if avg_loss < 0 else 0.0
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() < 0 else (np.inf if wins.sum() > 0 else 0.0)
    std = r.std(ddof=1) if n > 1 else 0.0
    sharpe = r.mean() / std if std > 0 else 0.0

    exit_counts = trades["exit_reason"].value_counts(normalize=True).to_dict()

    return BacktestStats(
        n_trades=n,
        n_wins=int((r > 0).sum()),
        n_losses=int((r < 0).sum()),
        win_rate=float((r > 0).mean()),
        avg_r=float(r.mean()),
        median_r=float(r.median()),
        total_r=float(r.sum()),
        max_dd_r=_max_drawdown(equity_r),
        expectancy=float(r.mean()),
        payoff_ratio=float(payoff),
        longest_win_streak=_longest_streak(r, lambda v: v > 0),
        longest_loss_streak=_longest_streak(r, lambda v: v < 0),
        profit_factor=float(profit_factor),
        sharpe_like=float(sharpe),
        bars_per_trade=float(trades["bars_held"].mean()),
        pct_exits_by_reason={k: float(v) for k, v in exit_counts.items()},
    )
