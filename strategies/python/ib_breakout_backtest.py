#!/usr/bin/env python3
"""
IB Breakout Extension Strategy - Python Backtest

Mirrors the Pine Script strategy in
strategies/pinescript/ib-breakout-extension-strategy.pine

Logic
-----
* Initial Balance = high/low of first hour of RTH (default 09:30-10:30 ET).
* After IB completes, arm two stop entries:
    - long  stop at IB high (first wick = breakout)
    - short stop at IB low  (first wick = breakdown)
  Whichever fires first becomes the trade.
* If the OPPOSITE side later wicks through, the day is a "double break"
  and the position is flattened (those days have much weaker stats).
* Scale out at the IB-range extensions: +/-0.2, 0.4, 0.6, 0.8, 1.0.
* Hard stop at the opposite side of the IB (configurable).
* Flatten N minutes before RTH close.

Usage
-----
    # Backtest a CSV (5-min RTH bars in ET)
    python ib_breakout_backtest.py --csv data/nq_5m.csv --symbol NQ

    # Run on synthetic data to validate the engine
    python ib_breakout_backtest.py --synthetic 750 --start-price 18000

CSV format (timestamps assumed already in America/New_York, tz-naive):
    timestamp,open,high,low,close,volume
    2023-01-03 09:30:00,11500.25,11512.00,11498.50,11505.75,1500
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    ib_start: time = time(9, 30)
    ib_end: time = time(10, 30)
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)

    # Targets: list of (multiplier, exit_pct_of_full_position)
    targets: Tuple[Tuple[float, float], ...] = (
        (0.2, 20.0),
        (0.4, 25.0),
        (0.6, 20.0),
        (0.8, 20.0),
        (1.0, 15.0),
    )

    stop_mode: str = "ib_opposite"  # ib_opposite | ib_mid | custom
    custom_stop_x: float = 0.5

    flatten_on_double_break: bool = True
    flatten_mins_before_close: int = 15

    # Range filter (% of price). 0 / large numbers = disabled.
    min_ib_range_pct: float = 0.0
    max_ib_range_pct: float = 5.0

    trade_longs: bool = True
    trade_shorts: bool = True

    # Slippage in price units applied adversely on stop fills only
    slippage_per_unit: float = 0.0

    # Sizing - fixed % of starting equity risked from entry to stop
    risk_per_trade_pct: float = 1.0
    starting_equity: float = 100_000.0


# -----------------------------------------------------------------------------
# Trade record
# -----------------------------------------------------------------------------

@dataclass
class Trade:
    date: pd.Timestamp
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    ib_high: float
    ib_low: float
    ib_range: float
    units: float
    stop_price: float
    legs: List[Tuple[int, float, float, pd.Timestamp, str]] = field(default_factory=list)
    is_double_break: bool = False

    @property
    def pnl(self) -> float:
        sign = 1 if self.side == "long" else -1
        return sum((leg[1] - self.entry_price) * sign * leg[2] for leg in self.legs)

    @property
    def r_multiple(self) -> float:
        risk_per_unit = abs(self.entry_price - self.stop_price)
        if risk_per_unit == 0 or self.units == 0:
            return 0.0
        return self.pnl / (risk_per_unit * self.units)


# -----------------------------------------------------------------------------
# Backtest engine
# -----------------------------------------------------------------------------

def run_backtest(df: pd.DataFrame, cfg: Config) -> Tuple[List[Trade], dict]:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["date"] = df.index.normalize()

    trades: List[Trade] = []
    day_class = {
        "single_long": 0,
        "single_short": 0,
        "double": 0,
        "none": 0,
        "skipped_filter": 0,
    }
    target_hits_long = [0] * len(cfg.targets)
    target_hits_short = [0] * len(cfg.targets)
    n_long_days = 0
    n_short_days = 0

    for date, day in df.groupby("date"):
        day = day[(day.index.time >= cfg.rth_start) & (day.index.time <= cfg.rth_end)]
        if day.empty:
            continue

        ib_bars = day[(day.index.time >= cfg.ib_start) & (day.index.time < cfg.ib_end)]
        if ib_bars.empty:
            continue

        ib_high = float(ib_bars["high"].max())
        ib_low = float(ib_bars["low"].min())
        ib_range = ib_high - ib_low
        if ib_range <= 0:
            continue

        post_ib = day[day.index.time >= cfg.ib_end]
        if post_ib.empty:
            continue

        # Range filter
        ref_price = float(day["open"].iloc[0])
        ib_range_pct = ib_range / ref_price * 100
        if ib_range_pct < cfg.min_ib_range_pct or ib_range_pct > cfg.max_ib_range_pct:
            day_class["skipped_filter"] += 1
            continue

        # Find first wick on either side
        above = post_ib[post_ib["high"] > ib_high]
        below = post_ib[post_ib["low"] < ib_low]
        first_above = above.index.min() if not above.empty else None
        first_below = below.index.min() if not below.empty else None

        # Day classification (independent of trade outcome)
        is_double_day = first_above is not None and first_below is not None
        if first_above is None and first_below is None:
            day_class["none"] += 1
            continue
        if is_double_day:
            day_class["double"] += 1
        elif first_above is not None:
            day_class["single_long"] += 1
        else:
            day_class["single_short"] += 1

        # Same bar wicks both sides -> ambiguous, skip the trade
        if first_above is not None and first_below is not None and first_above == first_below:
            continue

        if first_above is not None and (first_below is None or first_above < first_below):
            side = "long"
            entry_idx = first_above
            entry_price = ib_high + cfg.slippage_per_unit
            n_long_days += 1
        else:
            side = "short"
            entry_idx = first_below
            entry_price = ib_low - cfg.slippage_per_unit
            n_short_days += 1

        if (side == "long" and not cfg.trade_longs) or (
            side == "short" and not cfg.trade_shorts
        ):
            continue

        # Stop loss
        if cfg.stop_mode == "ib_opposite":
            stop_price = ib_low if side == "long" else ib_high
        elif cfg.stop_mode == "ib_mid":
            stop_price = (ib_high + ib_low) / 2
        else:
            stop_price = (
                ib_high - ib_range * cfg.custom_stop_x
                if side == "long"
                else ib_low + ib_range * cfg.custom_stop_x
            )

        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            continue

        units = (cfg.risk_per_trade_pct / 100 * cfg.starting_equity) / risk_per_unit

        target_prices = []
        for mult, pct in cfg.targets:
            tp = (
                ib_high + ib_range * mult
                if side == "long"
                else ib_low - ib_range * mult
            )
            target_prices.append((tp, pct / 100))

        trade = Trade(
            date=date,
            side=side,
            entry_time=entry_idx,
            entry_price=entry_price,
            ib_high=ib_high,
            ib_low=ib_low,
            ib_range=ib_range,
            units=units,
            stop_price=stop_price,
        )

        # Walk forward through bars at and after the entry bar
        post_entry = post_ib[post_ib.index >= entry_idx]
        remaining = units
        targets_hit = [False] * len(target_prices)
        broke_other = False

        close_minute_cutoff = (
            cfg.rth_end.hour * 60 + cfg.rth_end.minute - cfg.flatten_mins_before_close
        )

        for ts, bar in post_entry.iterrows():
            if remaining <= 1e-9:
                break
            high = float(bar["high"])
            low = float(bar["low"])

            # Detect double break (opposite side wicks through)
            if side == "long" and low < ib_low:
                broke_other = True
            if side == "short" and high > ib_high:
                broke_other = True

            # Pessimistic intra-bar order: stop first, then targets
            stopped = False
            if side == "long" and low <= stop_price:
                fill = stop_price - cfg.slippage_per_unit
                trade.legs.append((-1, fill, remaining, ts, "stop"))
                remaining = 0
                stopped = True
            elif side == "short" and high >= stop_price:
                fill = stop_price + cfg.slippage_per_unit
                trade.legs.append((-1, fill, remaining, ts, "stop"))
                remaining = 0
                stopped = True

            if not stopped:
                for i, (tp, pct) in enumerate(target_prices):
                    if targets_hit[i]:
                        continue
                    hit = (side == "long" and high >= tp) or (
                        side == "short" and low <= tp
                    )
                    if hit:
                        qty = min(units * pct, remaining)
                        trade.legs.append((i, tp, qty, ts, "target"))
                        remaining -= qty
                        targets_hit[i] = True
                        if side == "long":
                            target_hits_long[i] += 1
                        else:
                            target_hits_short[i] += 1

            if cfg.flatten_on_double_break and broke_other and remaining > 1e-9:
                trade.legs.append((-1, float(bar["close"]), remaining, ts, "double"))
                remaining = 0
                trade.is_double_break = True
                break

            ts_minute = ts.hour * 60 + ts.minute
            if ts_minute >= close_minute_cutoff and remaining > 1e-9:
                trade.legs.append((-1, float(bar["close"]), remaining, ts, "eod"))
                remaining = 0
                break

        if remaining > 1e-9:
            last_ts = post_entry.index[-1]
            last_close = float(post_entry["close"].iloc[-1])
            trade.legs.append((-1, last_close, remaining, last_ts, "eod"))

        trades.append(trade)

    summary = compute_summary(
        trades, day_class, target_hits_long, target_hits_short, n_long_days, n_short_days, cfg
    )
    return trades, summary


def compute_summary(
    trades, day_class, hits_long, hits_short, n_long, n_short, cfg
) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "day_classifications": day_class}

    pnls = np.array([t.pnl for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    equity_curve = np.cumsum(pnls) + cfg.starting_equity
    peak = np.maximum.accumulate(equity_curve)
    dd_pct = (peak - equity_curve) / peak * 100
    max_dd = float(dd_pct.max())

    total_pnl = float(pnls.sum())
    pf = (
        float(wins.sum() / abs(losses.sum()))
        if losses.size > 0 and losses.sum() != 0
        else float("inf")
    )

    return {
        "n_trades": n,
        "n_long_entries": n_long,
        "n_short_entries": n_short,
        "win_rate": float((pnls > 0).mean() * 100),
        "total_pnl": total_pnl,
        "expectancy": float(pnls.mean()),
        "expectancy_r": float(rs.mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": pf,
        "max_drawdown_pct": max_dd,
        "ending_equity": cfg.starting_equity + total_pnl,
        "day_classifications": day_class,
        "target_hits_long_pct": [
            (h / n_long * 100) if n_long else 0.0 for h in hits_long
        ],
        "target_hits_short_pct": [
            (h / n_short * 100) if n_short else 0.0 for h in hits_short
        ],
    }


# -----------------------------------------------------------------------------
# Synthetic data generator (for engine validation only)
# -----------------------------------------------------------------------------

def generate_synthetic_data(
    n_days: int = 750, start_price: float = 18000.0, seed: int = 42
) -> pd.DataFrame:
    """
    Generates plausible 5-minute RTH bars with day-level regimes calibrated
    to roughly match the published edgeful.com IB stats (~75% single break,
    ~21% double break, ~3% no break on the first hour).

    NOT a substitute for real data - only useful to validate that the
    backtest engine runs end-to-end with realistic-looking inputs.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 79  # 09:30..16:00 inclusive at 5 min = 79 bars
    ib_bars = 12  # 60 minutes / 5
    bar_minutes = 5

    daily_vol = 0.012
    bar_vol = daily_vol / np.sqrt(bars_per_day)
    bar_drift = 0.0003 / bars_per_day

    rows = []
    price = start_price
    current = pd.Timestamp("2023-01-03 09:30:00")

    for _ in range(n_days):
        while current.weekday() >= 5:
            current += pd.Timedelta(days=1)

        # Sample day regime to roughly match published rates
        u = rng.random()
        if u < 0.75:
            regime = "single"
            direction = 1 if rng.random() < 0.5 else -1
        elif u < 0.96:
            regime = "double"
            direction = 1 if rng.random() < 0.5 else -1
        else:
            regime = "none"
            direction = 0

        ib_open = price
        # Track IB so we can drive post-IB behavior consistently
        ib_high_running = price
        ib_low_running = price

        for bar_idx in range(bars_per_day):
            t_norm = bar_idx / bars_per_day
            vol_scale = 1.0 + 0.6 * (1 - 4 * (t_norm - 0.5) ** 2)
            sigma = bar_vol * vol_scale

            # IB phase: amplified vol so the IB range is meaningful
            if bar_idx < ib_bars:
                ret = rng.normal(0, sigma * 1.6)
                new_price = price * (1 + ret)
                wick_up = abs(rng.normal(0, sigma * 0.7)) * price
                wick_dn = abs(rng.normal(0, sigma * 0.7)) * price
                o = price
                c = new_price
                h = max(o, c) + wick_up
                l = min(o, c) - wick_dn
                ib_high_running = max(ib_high_running, h)
                ib_low_running = min(ib_low_running, l)
            else:
                bars_post = bar_idx - ib_bars
                ib_range = ib_high_running - ib_low_running

                if regime == "single":
                    drift = direction * sigma * 0.55
                    ret = rng.normal(drift, sigma * 0.6)
                elif regime == "double":
                    mid = (ib_high_running + ib_low_running) / 2
                    pull = (mid - price) / price * 0.25
                    ret = rng.normal(pull, sigma * 0.7)
                else:
                    mid = (ib_high_running + ib_low_running) / 2
                    pull = (mid - price) / price * 0.6
                    ret = rng.normal(pull, sigma * 0.4)

                new_price = price * (1 + ret)
                wick_up = abs(rng.normal(0, sigma * 0.4)) * price
                wick_dn = abs(rng.normal(0, sigma * 0.4)) * price

                o = price
                c = new_price
                h = max(o, c) + wick_up
                l = min(o, c) - wick_dn

                if regime == "single":
                    # Clamp so the opposite IB side is never crossed
                    if direction > 0:
                        l = max(l, ib_low_running + ib_range * 0.05)
                    else:
                        h = min(h, ib_high_running - ib_range * 0.05)
                elif regime == "none":
                    # Stay strictly inside the IB
                    h = min(h, ib_high_running - ib_range * 0.02)
                    l = max(l, ib_low_running + ib_range * 0.02)
                elif regime == "double":
                    # Force a wick above and below at scheduled bars
                    if bars_post == 2:
                        if direction > 0:
                            h = max(h, ib_high_running + ib_range * 0.25)
                            new_price = max(new_price, ib_high_running + ib_range * 0.05)
                        else:
                            l = min(l, ib_low_running - ib_range * 0.25)
                            new_price = min(new_price, ib_low_running - ib_range * 0.05)
                    if bars_post == 14:
                        if direction > 0:
                            l = min(l, ib_low_running - ib_range * 0.25)
                            new_price = min(new_price, ib_low_running - ib_range * 0.05)
                        else:
                            h = max(h, ib_high_running + ib_range * 0.25)
                            new_price = max(new_price, ib_high_running + ib_range * 0.05)

            ts = current + pd.Timedelta(minutes=bar_idx * bar_minutes)
            rows.append(
                {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1000}
            )
            price = new_price

        current = (current + pd.Timedelta(days=1)).normalize() + pd.Timedelta(
            hours=9, minutes=30
        )

    return pd.DataFrame(rows).set_index("timestamp")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def print_summary(summary: dict, symbol: str, n_bars: int, cfg: Config) -> None:
    print(f"\n{'=' * 72}")
    print(f"  IB Breakout Extension Backtest  -  {symbol}")
    print(f"  Bars processed: {n_bars:,}")
    print(f"{'=' * 72}\n")

    if summary["n_trades"] == 0:
        print("  No trades.")
        return

    dc = summary["day_classifications"]
    classified = dc["single_long"] + dc["single_short"] + dc["double"] + dc["none"]
    print("  Day classification (post-IB)")
    print(
        f"    single break (long) :  {dc['single_long']:5d}  "
        f"({dc['single_long'] / classified * 100:5.1f}%)"
    )
    print(
        f"    single break (short):  {dc['single_short']:5d}  "
        f"({dc['single_short'] / classified * 100:5.1f}%)"
    )
    print(
        f"    double break        :  {dc['double']:5d}  "
        f"({dc['double'] / classified * 100:5.1f}%)"
    )
    print(
        f"    no break            :  {dc['none']:5d}  "
        f"({dc['none'] / classified * 100:5.1f}%)"
    )
    print(f"    skipped (filter)    :  {dc['skipped_filter']:5d}\n")

    print("  Target hit rates (long breakouts)")
    for (mult, _), pct in zip(cfg.targets, summary["target_hits_long_pct"]):
        print(f"    +{mult:.1f} : {pct:5.1f}%")

    print("\n  Target hit rates (short breakdowns)")
    for (mult, _), pct in zip(cfg.targets, summary["target_hits_short_pct"]):
        print(f"    -{mult:.1f} : {pct:5.1f}%")

    print(f"\n  Trades        : {summary['n_trades']}")
    print(f"  Win rate      : {summary['win_rate']:.1f}%")
    print(f"  Total PnL     : ${summary['total_pnl']:,.2f}")
    print(f"  Expectancy    : ${summary['expectancy']:,.2f}/trade")
    print(f"  Expectancy R  : {summary['expectancy_r']:.3f}R")
    print(f"  Avg win       : ${summary['avg_win']:,.2f}")
    print(f"  Avg loss      : ${summary['avg_loss']:,.2f}")
    print(f"  Profit factor : {summary['profit_factor']:.2f}")
    print(f"  Max drawdown  : {summary['max_drawdown_pct']:.2f}%")
    print(f"  Ending equity : ${summary['ending_equity']:,.2f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="Path to OHLCV CSV (timestamp,open,high,low,close,volume)")
    g.add_argument("--synthetic", type=int, help="Generate N days of synthetic data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-price", type=float, default=18000.0)
    parser.add_argument("--symbol", default="SYN")
    parser.add_argument("--min-ib-pct", type=float, default=0.0)
    parser.add_argument("--max-ib-pct", type=float, default=5.0)
    parser.add_argument("--no-flatten-double", action="store_true")
    parser.add_argument("--longs-only", action="store_true")
    parser.add_argument("--shorts-only", action="store_true")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    else:
        df = generate_synthetic_data(
            n_days=args.synthetic, start_price=args.start_price, seed=args.seed
        )

    cfg = Config(
        min_ib_range_pct=args.min_ib_pct,
        max_ib_range_pct=args.max_ib_pct,
        flatten_on_double_break=not args.no_flatten_double,
        trade_longs=not args.shorts_only,
        trade_shorts=not args.longs_only,
    )

    _, summary = run_backtest(df, cfg)
    print_summary(summary, args.symbol, len(df), cfg)


if __name__ == "__main__":
    main()
