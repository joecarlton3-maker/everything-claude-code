"""Trade simulator — bar-by-bar execution matching Pine sections 7.5 + hit-detection.

KEY ASSUMPTIONS (documented for honesty about backtest fidelity):

1. ENTRY PRICE = close of the signal bar.
   Matches Pine. In live trading you'd fill on next bar's open — this is a
   known optimistic bias. For tuning relative comparisons across param sets
   it doesn't matter (bias is constant); for absolute PnL claims it does.

2. HIT DETECTION starts on the bar AFTER entry (`bar_index > tradeEntryBar`).
   Matches Pine. No same-bar entry+exit shenanigans.

3. SAME-BAR SL+TP CONFLICT: we pessimistically assume SL hits first.
   Pine doesn't address this (it checks both, sets both flags; the trade
   closes only when TP3 OR SL OR timeout triggers). To avoid optimistic
   bias, when SL and a TP both touch on the same bar, we route through SL
   for the un-hit portion. Already-hit lower TPs still count.

4. PARTIAL SCALE-OUT = even thirds at TP1/TP2/TP3.
   Matches Pine's `calcRealizedR`. Change this here if you want different
   weighting — keep it in sync with whatever you actually trade.

5. NEW SIGNAL WHILE IN TRADE = close existing trade at current bar close,
   book realized R based on whatever TPs were hit, then open new trade.
   (Pine just overwrites — silently loses the open P&L. We do the right
   thing.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco if not args else args[0]


# Exit reason codes (numba-friendly enum)
EXIT_TP3 = 1
EXIT_SL = 2
EXIT_TIMEOUT = 3
EXIT_SIGNAL_FLIP = 4
EXIT_END_OF_DATA = 5


@dataclass
class SimulationResult:
    trades: pd.DataFrame
    equity_r: pd.Series          # cumulative realized R per bar (open trades not marked-to-market)
    in_trade: pd.Series          # bool: was a trade open at end of this bar?


@njit(cache=True)
def _simulate_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    flip_up: np.ndarray,
    flip_down: np.ndarray,
    atr_eff: np.ndarray,
    last_pivot_high: np.ndarray,
    last_pivot_low: np.ndarray,
    tp1_r: np.ndarray,
    tp2_r: np.ndarray,
    tp3_r: np.ndarray,
    sl_atr_mult: float,
    trade_max_age: int,
    warmup: int,
):
    n = high.shape[0]

    # Preallocate trade arrays (max n_bars / 2 trades is a generous upper bound)
    max_trades = max(n // 2, 16)
    t_entry_bar = np.full(max_trades, -1, dtype=np.int64)
    t_exit_bar = np.full(max_trades, -1, dtype=np.int64)
    t_direction = np.zeros(max_trades, dtype=np.int8)
    t_entry = np.full(max_trades, np.nan)
    t_sl = np.full(max_trades, np.nan)
    t_tp1 = np.full(max_trades, np.nan)
    t_tp2 = np.full(max_trades, np.nan)
    t_tp3 = np.full(max_trades, np.nan)
    t_tp1r = np.full(max_trades, np.nan)
    t_tp2r = np.full(max_trades, np.nan)
    t_tp3r = np.full(max_trades, np.nan)
    t_hit_tp1 = np.zeros(max_trades, dtype=np.bool_)
    t_hit_tp2 = np.zeros(max_trades, dtype=np.bool_)
    t_hit_tp3 = np.zeros(max_trades, dtype=np.bool_)
    t_realized_r = np.zeros(max_trades, dtype=np.float64)
    t_exit_reason = np.zeros(max_trades, dtype=np.int8)
    n_trades = 0

    # Active trade state (only one open at a time)
    active = False
    a_idx = -1                # index into t_* arrays

    equity_r = np.zeros(n, dtype=np.float64)
    in_trade_out = np.zeros(n, dtype=np.bool_)

    cum_r = 0.0

    for i in range(n):
        if i < warmup:
            equity_r[i] = cum_r
            continue

        # ── 1. Check exits on the active trade (next bar after entry) ──
        if active and i > t_entry_bar[a_idx]:
            dir_ = t_direction[a_idx]
            sl = t_sl[a_idx]
            tp1 = t_tp1[a_idx]
            tp2 = t_tp2[a_idx]
            tp3 = t_tp3[a_idx]
            tp1r = t_tp1r[a_idx]
            tp2r = t_tp2r[a_idx]
            tp3r = t_tp3r[a_idx]

            if dir_ == 1:
                tp1_reached = high[i] >= tp1
                tp2_reached = high[i] >= tp2
                tp3_reached = high[i] >= tp3
                sl_hit = low[i] <= sl
            else:
                tp1_reached = low[i] <= tp1
                tp2_reached = low[i] <= tp2
                tp3_reached = low[i] <= tp3
                sl_hit = high[i] >= sl

            # Update TP-hit flags (sticky)
            if tp1_reached and not t_hit_tp1[a_idx]:
                t_hit_tp1[a_idx] = True
            if tp2_reached and not t_hit_tp2[a_idx]:
                t_hit_tp2[a_idx] = True
            if tp3_reached and not t_hit_tp3[a_idx]:
                t_hit_tp3[a_idx] = True

            trade_age = i - t_entry_bar[a_idx]
            timeout = trade_age >= trade_max_age

            # Resolve exit. If SL hit on same bar as a TP, pessimistically take SL.
            close_trade = False
            exit_reason = 0
            if t_hit_tp3[a_idx]:
                # All TPs filled — book full 3-tranche R
                realized = (tp1r + tp2r + tp3r) / 3.0
                close_trade = True
                exit_reason = EXIT_TP3
            elif sl_hit:
                # Already-hit TPs keep their R; remaining tranches take -1R.
                taken = 0.0
                remaining = 1.0
                if t_hit_tp1[a_idx]:
                    taken += (1.0 / 3.0) * tp1r
                    remaining -= 1.0 / 3.0
                if t_hit_tp2[a_idx]:
                    taken += (1.0 / 3.0) * tp2r
                    remaining -= 1.0 / 3.0
                realized = taken + remaining * (-1.0)
                close_trade = True
                exit_reason = EXIT_SL
            elif timeout:
                # Force close — already-hit TPs count, remaining tranches book 0R.
                taken = 0.0
                if t_hit_tp1[a_idx]:
                    taken += (1.0 / 3.0) * tp1r
                if t_hit_tp2[a_idx]:
                    taken += (1.0 / 3.0) * tp2r
                if t_hit_tp3[a_idx]:
                    taken += (1.0 / 3.0) * tp3r
                realized = taken
                close_trade = True
                exit_reason = EXIT_TIMEOUT
            else:
                realized = 0.0

            if close_trade:
                t_realized_r[a_idx] = realized
                t_exit_bar[a_idx] = i
                t_exit_reason[a_idx] = exit_reason
                cum_r += realized
                active = False
                a_idx = -1

        # ── 2. New signal handling ──
        new_long = flip_up[i]
        new_short = flip_down[i]

        if (new_long or new_short) and active:
            # Close existing trade at current bar's close, book realized R.
            dir_ = t_direction[a_idx]
            entry = t_entry[a_idx]
            tp1r = t_tp1r[a_idx]
            tp2r = t_tp2r[a_idx]
            tp3r = t_tp3r[a_idx]
            # Realized R from already-hit TPs + the remaining tranche at current close
            taken = 0.0
            tranches_filled = 0
            if t_hit_tp1[a_idx]:
                taken += (1.0 / 3.0) * tp1r
                tranches_filled += 1
            if t_hit_tp2[a_idx]:
                taken += (1.0 / 3.0) * tp2r
                tranches_filled += 1
            if t_hit_tp3[a_idx]:
                taken += (1.0 / 3.0) * tp3r
                tranches_filled += 1
            remaining = 1.0 - (tranches_filled / 3.0)
            # Mark remaining tranche to current close as R-multiple
            # R = (close - entry) / risk_per_unit, but we stored prices not risk.
            # Compute risk = |entry - sl| / 1R. We know tp1 = entry + risk * tp1r,
            # so risk = (tp1 - entry) / tp1r (long) or (entry - tp1) / tp1r (short).
            if dir_ == 1:
                risk = (t_tp1[a_idx] - entry) / max(tp1r, 1e-9)
                mtm_r = (close[i] - entry) / max(risk, 1e-9)
            else:
                risk = (entry - t_tp1[a_idx]) / max(tp1r, 1e-9)
                mtm_r = (entry - close[i]) / max(risk, 1e-9)
            realized = taken + remaining * mtm_r
            t_realized_r[a_idx] = realized
            t_exit_bar[a_idx] = i
            t_exit_reason[a_idx] = EXIT_SIGNAL_FLIP
            cum_r += realized
            active = False
            a_idx = -1

        if new_long and n_trades < max_trades:
            entry = close[i]
            sl_base = last_pivot_low[i] if not np.isnan(last_pivot_low[i]) else low[i]
            raw_sl = sl_base - sl_atr_mult * atr_eff[i]
            min_sl = entry - sl_atr_mult * atr_eff[i]
            sl = min(raw_sl, min_sl)
            risk = entry - sl
            if risk > 0:
                idx = n_trades
                t_entry_bar[idx] = i
                t_direction[idx] = 1
                t_entry[idx] = entry
                t_sl[idx] = sl
                t_tp1[idx] = entry + risk * tp1_r[i]
                t_tp2[idx] = entry + risk * tp2_r[i]
                t_tp3[idx] = entry + risk * tp3_r[i]
                t_tp1r[idx] = tp1_r[i]
                t_tp2r[idx] = tp2_r[i]
                t_tp3r[idx] = tp3_r[i]
                n_trades += 1
                a_idx = idx
                active = True
        elif new_short and n_trades < max_trades:
            entry = close[i]
            sl_base = last_pivot_high[i] if not np.isnan(last_pivot_high[i]) else high[i]
            raw_sl = sl_base + sl_atr_mult * atr_eff[i]
            min_sl = entry + sl_atr_mult * atr_eff[i]
            sl = max(raw_sl, min_sl)
            risk = sl - entry
            if risk > 0:
                idx = n_trades
                t_entry_bar[idx] = i
                t_direction[idx] = -1
                t_entry[idx] = entry
                t_sl[idx] = sl
                t_tp1[idx] = entry - risk * tp1_r[i]
                t_tp2[idx] = entry - risk * tp2_r[i]
                t_tp3[idx] = entry - risk * tp3_r[i]
                t_tp1r[idx] = tp1_r[i]
                t_tp2r[idx] = tp2_r[i]
                t_tp3r[idx] = tp3_r[i]
                n_trades += 1
                a_idx = idx
                active = True

        equity_r[i] = cum_r
        in_trade_out[i] = active

    # If still in a trade at end of data, close it at last close.
    if active:
        dir_ = t_direction[a_idx]
        entry = t_entry[a_idx]
        tp1r = t_tp1r[a_idx]
        tp2r = t_tp2r[a_idx]
        tp3r = t_tp3r[a_idx]
        taken = 0.0
        tranches_filled = 0
        if t_hit_tp1[a_idx]:
            taken += (1.0 / 3.0) * tp1r
            tranches_filled += 1
        if t_hit_tp2[a_idx]:
            taken += (1.0 / 3.0) * tp2r
            tranches_filled += 1
        if t_hit_tp3[a_idx]:
            taken += (1.0 / 3.0) * tp3r
            tranches_filled += 1
        remaining = 1.0 - (tranches_filled / 3.0)
        if dir_ == 1:
            risk = (t_tp1[a_idx] - entry) / max(tp1r, 1e-9)
            mtm_r = (close[n - 1] - entry) / max(risk, 1e-9)
        else:
            risk = (entry - t_tp1[a_idx]) / max(tp1r, 1e-9)
            mtm_r = (entry - close[n - 1]) / max(risk, 1e-9)
        realized = taken + remaining * mtm_r
        t_realized_r[a_idx] = realized
        t_exit_bar[a_idx] = n - 1
        t_exit_reason[a_idx] = EXIT_END_OF_DATA
        cum_r += realized
        equity_r[n - 1] = cum_r

    return (
        t_entry_bar[:n_trades],
        t_exit_bar[:n_trades],
        t_direction[:n_trades],
        t_entry[:n_trades],
        t_sl[:n_trades],
        t_tp1[:n_trades],
        t_tp2[:n_trades],
        t_tp3[:n_trades],
        t_tp1r[:n_trades],
        t_tp2r[:n_trades],
        t_tp3r[:n_trades],
        t_hit_tp1[:n_trades],
        t_hit_tp2[:n_trades],
        t_hit_tp3[:n_trades],
        t_realized_r[:n_trades],
        t_exit_reason[:n_trades],
        equity_r,
        in_trade_out,
    )


_EXIT_NAMES = {
    EXIT_TP3: "tp3",
    EXIT_SL: "sl",
    EXIT_TIMEOUT: "timeout",
    EXIT_SIGNAL_FLIP: "signal_flip",
    EXIT_END_OF_DATA: "end_of_data",
}


def simulate(
    *,
    df: pd.DataFrame,
    flip_up: pd.Series,
    flip_down: pd.Series,
    atr_eff: pd.Series,
    last_pivot_high: pd.Series,
    last_pivot_low: pd.Series,
    tp1_r: pd.Series,
    tp2_r: pd.Series,
    tp3_r: pd.Series,
    sl_atr_mult: float,
    trade_max_age: int,
    warmup: int,
) -> SimulationResult:
    """Run the trade simulator. Returns trade log + equity curve."""
    res = _simulate_numba(
        high=df["high"].values.astype(np.float64),
        low=df["low"].values.astype(np.float64),
        close=df["close"].values.astype(np.float64),
        flip_up=flip_up.values.astype(np.bool_),
        flip_down=flip_down.values.astype(np.bool_),
        atr_eff=atr_eff.values.astype(np.float64),
        last_pivot_high=last_pivot_high.values.astype(np.float64),
        last_pivot_low=last_pivot_low.values.astype(np.float64),
        tp1_r=tp1_r.values.astype(np.float64),
        tp2_r=tp2_r.values.astype(np.float64),
        tp3_r=tp3_r.values.astype(np.float64),
        sl_atr_mult=sl_atr_mult,
        trade_max_age=trade_max_age,
        warmup=warmup,
    )
    (entry_bar, exit_bar, direction, entry, sl, tp1, tp2, tp3,
     tp1r, tp2r, tp3r, hit_tp1, hit_tp2, hit_tp3, realized_r,
     exit_reason, equity_r, in_trade_arr) = res

    idx = df.index
    trades_df = pd.DataFrame({
        "entry_time": idx[entry_bar] if len(entry_bar) else pd.DatetimeIndex([]),
        "exit_time": idx[exit_bar] if len(exit_bar) else pd.DatetimeIndex([]),
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "bars_held": exit_bar - entry_bar,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp1_r": tp1r,
        "tp2_r": tp2r,
        "tp3_r": tp3r,
        "hit_tp1": hit_tp1,
        "hit_tp2": hit_tp2,
        "hit_tp3": hit_tp3,
        "realized_r": realized_r,
        "exit_reason": [_EXIT_NAMES.get(int(r), "unknown") for r in exit_reason],
    })

    return SimulationResult(
        trades=trades_df,
        equity_r=pd.Series(equity_r, index=idx),
        in_trade=pd.Series(in_trade_arr, index=idx),
    )
