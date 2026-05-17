#!/usr/bin/env python3
"""Anchored walk-forward CLI.

Re-uses the same grid as run_sweep.py but evaluates per-fold OOS.

Output: per-fold winner params + train vs test objective.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sats.config import SatsConfig  # noqa: E402
from sats.data.synthetic import synthetic_ohlcv  # noqa: E402
from sats.tuning import anchored_walk_forward  # noqa: E402

# Reuse the sweep grid
sys.path.insert(0, str(ROOT / "scripts"))
from run_sweep import build_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-source", choices=["synthetic", "databento"], default="synthetic")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeframe-minutes", type=int, default=15)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--symbol", default="ES.c.0")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--objective", default="calmar", choices=["calmar", "total_r", "sharpe_like"])
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--output", type=Path, default=Path("reports/walk_forward.csv"))
    return p.parse_args()


def load_data(args):
    if args.data_source == "synthetic":
        return synthetic_ohlcv(
            n_bars=args.bars, seed=args.seed,
            freq=f"{args.timeframe_minutes}min",
        )
    from sats.data.databento_loader import load_from_databento, resample_ohlcv
    raw = load_from_databento(args.symbol, args.start, args.end, schema="ohlcv-1m")
    return resample_ohlcv(raw, f"{args.timeframe_minutes}min")


def main() -> int:
    args = parse_args()
    df = load_data(args)
    grid = build_grid()
    # IMPORTANT: preset="Custom" — see comment in run_sweep.py
    base_cfg = SatsConfig(timeframe_minutes=args.timeframe_minutes, preset="Custom")

    print(f"Walk-forward: {args.folds} folds × {len(grid)} candidates × {len(df):,} bars")
    print(f"Objective: {args.objective}\n")

    result = anchored_walk_forward(
        df, base_cfg, grid,
        n_folds=args.folds,
        objective=args.objective,
        n_jobs=args.jobs,
        verbose=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(args.output, index=False)
    print(f"\nResults → {args.output}\n")

    # Pretty per-fold print
    print("Per-fold winners:")
    print("─" * 100)
    for f in result.folds:
        print(f"\nFold {f.fold_idx}: train [{f.train_start.date()} → {f.train_end.date()}]  "
              f"test [{f.test_start.date()} → {f.test_end.date()}]")
        print(f"  Winner params: {f.winner_params}")
        print(f"  Train: obj={f.train_objective:+.3f}  trades={f.train_stats['n_trades']:>4}  "
              f"WR={f.train_stats['win_rate']:.1%}  total_R={f.train_stats['total_r']:+.2f}")
        print(f"  Test : obj={f.test_objective:+.3f}  trades={f.test_stats['n_trades']:>4}  "
              f"WR={f.test_stats['win_rate']:.1%}  total_R={f.test_stats['total_r']:+.2f}")

    print("\n" + "═" * 100)
    print(f"Mean OOS objective across {args.folds} folds: {result.mean_test_objective:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
