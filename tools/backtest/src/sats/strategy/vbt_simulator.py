"""vectorbt-based single-target simulator.

Used when SatsConfig.exit_mode == "single_target". Full position closes at
TP1; TP2/TP3 are ignored. Wraps vectorbt's `Portfolio.from_signals` with
per-bar SL/TP percentages so each entry uses the SL/TP computed at its own
bar (not a single global stop).

Output is mapped back into the same `SimulationResult` shape as the
three-target path so downstream code (stats, sweep, walk-forward) doesn't
care which engine ran.

KNOWN LIMITATION — SL/TP precision:
    vectorbt.Portfolio.from_signals checks stops against `close` only.
    A bar that gaps through the SL exits at the gapped close, not at the
    stop price — so realized R can exceed -1R on bad gaps. The numba
    simulator uses OHLC and is precise to the bar. For accurate intrabar
    fills, use exit_mode="three_target" (or "single_target" with a future
    OHLC-aware vbt implementation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vectorbt as vbt

from .simulator import SimulationResult


# vectorbt's stop-direction conventions:
#   sl_stop > 0  → close drops by sl_stop * entry_price triggers SL (for longs)
#   tp_stop > 0  → close rises by tp_stop * entry_price triggers TP (for longs)
# Shorts mirror automatically.


def _compute_entry_levels(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    flip_up: pd.Series,
    flip_down: pd.Series,
    atr_eff: pd.Series,
    last_pivot_high: pd.Series,
    last_pivot_low: pd.Series,
    tp1_r: pd.Series,
    sl_atr_mult: float,
):
    """Compute per-bar SL and TP1 prices AS IF an entry happened at this bar.

    Mirrors the Pine logic in simulator.py:
        long:   sl = min(slBase - sl_mult*atr, entry - sl_mult*atr)
        short:  sl = max(slBase + sl_mult*atr, entry + sl_mult*atr)
    where slBase falls back to the bar's low/high when no pivot is available.

    Returns SL/TP1 percentages (positive) suitable for vectorbt's sl_stop/tp_stop.
    """
    entry = close

    # Long-side SL/TP
    long_sl_base = last_pivot_low.fillna(low)
    long_raw_sl = long_sl_base - sl_atr_mult * atr_eff
    long_min_sl = entry - sl_atr_mult * atr_eff
    long_sl_price = np.minimum(long_raw_sl, long_min_sl)
    long_risk = (entry - long_sl_price).clip(lower=0)
    long_tp_price = entry + long_risk * tp1_r

    # Short-side SL/TP
    short_sl_base = last_pivot_high.fillna(high)
    short_raw_sl = short_sl_base + sl_atr_mult * atr_eff
    short_min_sl = entry + sl_atr_mult * atr_eff
    short_sl_price = np.maximum(short_raw_sl, short_min_sl)
    short_risk = (short_sl_price - entry).clip(lower=0)
    short_tp_price = entry - short_risk * tp1_r

    # Convert to fractional stops (vectorbt wants positive fractions of entry)
    long_sl_pct = (entry - long_sl_price) / entry.replace(0, np.nan)
    long_tp_pct = (long_tp_price - entry) / entry.replace(0, np.nan)
    short_sl_pct = (short_sl_price - entry) / entry.replace(0, np.nan)
    short_tp_pct = (entry - short_tp_price) / entry.replace(0, np.nan)

    return {
        "long_sl_pct": long_sl_pct,
        "long_tp_pct": long_tp_pct,
        "long_risk": long_risk,
        "long_sl_price": long_sl_price,
        "long_tp_price": long_tp_price,
        "short_sl_pct": short_sl_pct,
        "short_tp_pct": short_tp_pct,
        "short_risk": short_risk,
        "short_sl_price": short_sl_price,
        "short_tp_price": short_tp_price,
    }


def simulate_vbt_single_target(
    *,
    df: pd.DataFrame,
    flip_up: pd.Series,
    flip_down: pd.Series,
    atr_eff: pd.Series,
    last_pivot_high: pd.Series,
    last_pivot_low: pd.Series,
    tp1_r: pd.Series,
    sl_atr_mult: float,
    trade_max_age: int,
    warmup: int,
) -> SimulationResult:
    """Run single-target backtest via vectorbt.

    Returns a SimulationResult shaped identically to the three-target engine,
    so downstream stats/sweep code is engine-agnostic.
    """
    n = len(df)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Zero out signals during warmup so vectorbt doesn't trade on cold state
    warm_mask = np.arange(n) >= warmup
    entries_long = flip_up & warm_mask
    entries_short = flip_down & warm_mask

    levels = _compute_entry_levels(
        close=close, high=high, low=low,
        flip_up=flip_up, flip_down=flip_down,
        atr_eff=atr_eff,
        last_pivot_high=last_pivot_high, last_pivot_low=last_pivot_low,
        tp1_r=tp1_r,
        sl_atr_mult=sl_atr_mult,
    )

    # Build per-bar stop arrays. vectorbt applies the value at the entry bar.
    # We feed both long and short variants and let vbt pick based on which signal fires.
    # When both long and short would fire on the same bar, longs win (matches Pine).

    # Combine: prefer long if both fire same bar (rare given supertrend flips)
    short_only = entries_short & ~entries_long
    long_only = entries_long

    sl_pct = pd.Series(np.nan, index=close.index)
    tp_pct = pd.Series(np.nan, index=close.index)
    sl_pct[long_only] = levels["long_sl_pct"][long_only]
    tp_pct[long_only] = levels["long_tp_pct"][long_only]
    sl_pct[short_only] = levels["short_sl_pct"][short_only]
    tp_pct[short_only] = levels["short_tp_pct"][short_only]

    # vectorbt rejects negative stops; clip with a tiny floor.
    sl_pct = sl_pct.clip(lower=1e-9)
    tp_pct = tp_pct.clip(lower=1e-9)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=long_only,
        short_entries=short_only,
        sl_stop=sl_pct,
        tp_stop=tp_pct,
        accumulate=False,         # new entry closes existing position
        upon_opposite_entry="ReverseReduce",
        max_size=1.0,
        size=1.0,                  # unit position — we work in R, not $
        size_type="amount",
        freq=pd.infer_freq(close.index) or "15min",
    )

    # Extract trades and rebuild a SimulationResult-shaped output.
    trades_records = pf.trades.records_readable
    if len(trades_records) == 0:
        empty = pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_bar", "exit_bar", "bars_held",
            "direction", "entry", "sl", "tp1", "tp2", "tp3",
            "tp1_r", "tp2_r", "tp3_r",
            "hit_tp1", "hit_tp2", "hit_tp3",
            "realized_r", "exit_reason",
        ])
        return SimulationResult(
            trades=empty,
            equity_r=pd.Series(0.0, index=close.index),
            in_trade=pd.Series(False, index=close.index),
        )

    trades_df = _build_trade_log(
        trades_records=trades_records,
        levels=levels,
        long_only=long_only,
        short_only=short_only,
        close=close,
        tp1_r_series=tp1_r,
        trade_max_age=trade_max_age,
    )

    equity_r = trades_df["realized_r"].cumsum()
    equity_curve = pd.Series(0.0, index=close.index)
    for i, t in trades_df.iterrows():
        equity_curve.loc[t["exit_time"]:] = equity_r.iloc[i]

    in_trade = pd.Series(False, index=close.index)
    for _, t in trades_df.iterrows():
        in_trade.loc[t["entry_time"]:t["exit_time"]] = True

    return SimulationResult(
        trades=trades_df,
        equity_r=equity_curve,
        in_trade=in_trade,
    )


def _build_trade_log(
    *,
    trades_records: pd.DataFrame,
    levels: dict,
    long_only: pd.Series,
    short_only: pd.Series,
    close: pd.Series,
    tp1_r_series: pd.Series,
    trade_max_age: int,
) -> pd.DataFrame:
    """Convert vectorbt trade records to our standard SimulationResult format.

    vectorbt 1.x `records_readable` columns:
        'Exit Trade Id', 'Column', 'Size', 'Entry Timestamp', 'Avg Entry Price',
        'Entry Fees', 'Exit Timestamp', 'Avg Exit Price', 'Exit Fees',
        'PnL', 'Return', 'Direction', 'Status', 'Position Id'
    """
    idx = close.index
    rows = []

    for _, rec in trades_records.iterrows():
        entry_time = pd.Timestamp(rec["Entry Timestamp"])
        exit_time = pd.Timestamp(rec["Exit Timestamp"])
        # Map timestamps back to integer bar indices
        entry_bar = idx.get_loc(entry_time)
        exit_bar = idx.get_loc(exit_time)
        if isinstance(entry_bar, slice):
            entry_bar = entry_bar.start
        if isinstance(exit_bar, slice):
            exit_bar = exit_bar.start
        direction = 1 if str(rec["Direction"]).lower().startswith("l") else -1
        entry_price = float(rec["Avg Entry Price"])
        exit_price = float(rec["Avg Exit Price"])

        if direction == 1:
            sl_price = float(levels["long_sl_price"].iloc[entry_bar])
            tp1_price = float(levels["long_tp_price"].iloc[entry_bar])
            risk = float(levels["long_risk"].iloc[entry_bar])
            pnl_price = exit_price - entry_price
        else:
            sl_price = float(levels["short_sl_price"].iloc[entry_bar])
            tp1_price = float(levels["short_tp_price"].iloc[entry_bar])
            risk = float(levels["short_risk"].iloc[entry_bar])
            pnl_price = entry_price - exit_price

        realized_r = pnl_price / risk if risk > 0 else 0.0
        tp1_r_val = float(tp1_r_series.iloc[entry_bar])

        bars_held = exit_bar - entry_bar
        if realized_r >= tp1_r_val - 1e-6:
            exit_reason = "tp1"
        elif realized_r <= -1.0 + 1e-6:
            exit_reason = "sl"
        elif bars_held >= trade_max_age:
            exit_reason = "timeout"
        else:
            exit_reason = "signal_flip"

        rows.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "bars_held": bars_held,
            "direction": direction,
            "entry": entry_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": np.nan,
            "tp3": np.nan,
            "tp1_r": tp1_r_val,
            "tp2_r": np.nan,
            "tp3_r": np.nan,
            "hit_tp1": realized_r >= tp1_r_val - 1e-6,
            "hit_tp2": False,
            "hit_tp3": False,
            "realized_r": realized_r,
            "exit_reason": exit_reason,
        })

    return pd.DataFrame(rows)
