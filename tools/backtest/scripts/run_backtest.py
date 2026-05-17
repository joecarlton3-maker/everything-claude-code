#!/usr/bin/env python3
"""Single-backtest CLI.

Usage:
    python scripts/run_backtest.py                       # synthetic data, defaults
    python scripts/run_backtest.py --bars 5000 --seed 42
    python scripts/run_backtest.py --tp-mode Dynamic --quality-strength 0.6

Use --data-source databento if you have DATABENTO_API_KEY set; the loader
will fetch and cache automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable without an install step.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sats.backtest.runner import run_backtest  # noqa: E402
from sats.config import SatsConfig  # noqa: E402
from sats.data.synthetic import synthetic_ohlcv  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SATS backtest runner")
    p.add_argument("--data-source", choices=["synthetic", "databento"], default="synthetic")
    p.add_argument("--symbol", default="ES.c.0", help="for --data-source databento")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-06-30")
    p.add_argument("--timeframe-minutes", type=int, default=15)
    # Synthetic-only knobs
    p.add_argument("--bars", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vol", type=float, default=0.5)
    p.add_argument("--drift", type=float, default=0.0)
    # Strategy knobs (most-tuned subset)
    p.add_argument("--preset", default="Auto")
    p.add_argument("--tp-mode", choices=["Fixed", "Dynamic"], default="Fixed")
    p.add_argument("--quality-strength", type=float, default=0.4)
    p.add_argument("--base-mult", type=float, default=2.0)
    p.add_argument("--sl-atr-mult", type=float, default=1.5)
    p.add_argument("--tp1-r", type=float, default=1.0)
    p.add_argument("--tp2-r", type=float, default=2.0)
    p.add_argument("--tp3-r", type=float, default=3.0)
    # Output
    p.add_argument("--save-trades", type=Path, help="write trades CSV to this path")
    p.add_argument("--json", action="store_true", help="emit stats as JSON only")
    return p.parse_args()


def load_data(args: argparse.Namespace):
    if args.data_source == "synthetic":
        return synthetic_ohlcv(
            n_bars=args.bars,
            seed=args.seed,
            vol=args.vol,
            drift=args.drift,
            freq=f"{args.timeframe_minutes}min",
        )
    # databento path
    from sats.data.databento_loader import load_from_databento, resample_ohlcv
    raw = load_from_databento(args.symbol, args.start, args.end, schema="ohlcv-1m")
    return resample_ohlcv(raw, f"{args.timeframe_minutes}min")


def main() -> int:
    args = parse_args()
    df = load_data(args)

    cfg = SatsConfig(
        timeframe_minutes=args.timeframe_minutes,
        preset=args.preset,
        tp_mode=args.tp_mode,
        quality_strength=args.quality_strength,
        base_mult=args.base_mult,
        sl_atr_mult=args.sl_atr_mult,
        tp1_r=args.tp1_r,
        tp2_r=args.tp2_r,
        tp3_r=args.tp3_r,
    )

    result = run_backtest(df, cfg)
    stats = result.stats.to_dict()

    if args.json:
        print(json.dumps(stats, indent=2, default=str))
    else:
        print(f"\n=== SATS Backtest — {args.data_source} ===")
        print(f"  Bars: {len(df):,}    Timeframe: {args.timeframe_minutes}m    Preset: {cfg.preset} → {result.bar_state.warmup_bars} warmup bars")
        print(f"  TP mode: {args.tp_mode}    quality_strength: {args.quality_strength}")
        print(f"\n  N trades: {stats['n_trades']}    Wins: {stats['n_wins']}    Losses: {stats['n_losses']}")
        print(f"  Win rate: {stats['win_rate']*100:5.1f}%    Avg R: {stats['avg_r']:+.3f}    Total R: {stats['total_r']:+.2f}")
        print(f"  Max DD (R): {stats['max_dd_r']:.2f}    Profit factor: {stats['profit_factor']:.2f}    Sharpe-like: {stats['sharpe_like']:+.2f}")
        print(f"  Payoff ratio: {stats['payoff_ratio']:.2f}    Bars/trade: {stats['bars_per_trade']:.1f}")
        print(f"  Streaks W/L: {stats['longest_win_streak']} / {stats['longest_loss_streak']}")
        print(f"  Exits: {stats['pct_exits_by_reason']}")

    if args.save_trades:
        args.save_trades.parent.mkdir(parents=True, exist_ok=True)
        result.simulation.trades.to_csv(args.save_trades, index=False)
        print(f"\n  Trades written: {args.save_trades}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
