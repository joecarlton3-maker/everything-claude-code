"""End-to-end backtest runner.

Single call site: `run_backtest(df, cfg)` → BacktestResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import SatsConfig, resolve
from ..indicator.engine import BarState, compute_bar_state
from ..strategy.pivots import last_pivot, pivot_high, pivot_low
from ..strategy.simulator import SimulationResult, simulate
from ..strategy.stats import BacktestStats, summarize


@dataclass
class BacktestResult:
    bar_state: BarState
    simulation: SimulationResult
    stats: BacktestStats


def run_backtest(
    df: pd.DataFrame,
    cfg: SatsConfig,
    *,
    pivot_left: int = 3,
    pivot_right: int = 3,
) -> BacktestResult:
    """Run the full pipeline.

    Pivot lookback defaults match Pine's `pivotLenInput = 3` (symmetric).
    """
    state = compute_bar_state(df, cfg)
    rc = resolve(cfg)

    ph = pivot_high(df["high"], pivot_left, pivot_right)
    pl = pivot_low(df["low"], pivot_left, pivot_right)
    last_ph = last_pivot(ph)
    last_pl = last_pivot(pl)

    sim = simulate(
        df=df,
        flip_up=state.supertrend.flip_up,
        flip_down=state.supertrend.flip_down,
        atr_eff=state.atr_eff,
        last_pivot_high=last_ph,
        last_pivot_low=last_pl,
        tp1_r=state.dynamic_tp.tp1_r,
        tp2_r=state.dynamic_tp.tp2_r,
        tp3_r=state.dynamic_tp.tp3_r,
        sl_atr_mult=rc.sl_atr_mult,
        trade_max_age=cfg.trade_max_age,
        warmup=state.warmup_bars,
    )

    stats = summarize(sim.trades, sim.equity_r)
    return BacktestResult(bar_state=state, simulation=sim, stats=stats)
