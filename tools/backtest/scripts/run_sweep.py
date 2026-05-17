#!/usr/bin/env python3
"""Parameter sweep CLI.

Runs the default 6-parameter ~432-combo grid:
    quality_strength × quality_curve × base_mult × sl_atr_mult
    × tp_profile × char_flip_min_age

Usage:
    python scripts/run_sweep.py                              # synthetic data
    python scripts/run_sweep.py --bars 20000 --jobs 8
    python scripts/run_sweep.py --top 30 --objective sharpe_like

Output: ranked CSV + top-N to stdout.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sats.config import SatsConfig  # noqa: E402
from sats.data.synthetic import synthetic_ohlcv  # noqa: E402
from sats.tuning import expand_grid, grid_search  # noqa: E402


# Default grid — picked for high-leverage params, kept under 500 combos.
DEFAULT_GRID = {
    "quality_strength": [0.0, 0.3, 0.5, 0.7],         # 4
    "quality_curve":    [1.0, 1.5, 2.0],              # 3
    "base_mult":        [1.5, 2.0, 2.5],              # 3
    "sl_atr_mult":      [1.0, 1.5, 2.0],              # 3
    "char_flip_min_age": [3, 5],                       # 2
    # TP profile encoded as a triple of R-multiples
    # (tp1, tp2, tp3) — handled below
}

TP_PROFILES = [
    (1.0, 2.0, 3.0),    # base
    (0.75, 1.5, 3.0),   # aggressive TP1
    # (1.5, 3.0, 5.0),  # wide — uncomment for larger sweep
]


def build_grid() -> list[dict]:
    base = expand_grid(DEFAULT_GRID)
    out = []
    for combo in base:
        for tp1, tp2, tp3 in TP_PROFILES:
            c = dict(combo)
            c["tp1_r"] = tp1
            c["tp2_r"] = tp2
            c["tp3_r"] = tp3
            out.append(c)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-source", choices=["synthetic", "databento"], default="synthetic")
    p.add_argument("--bars", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeframe-minutes", type=int, default=15)
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--symbol", default="ES.c.0")
    p.add_argument("--objective", default="calmar", choices=["calmar", "total_r", "sharpe_like"])
    p.add_argument("--jobs", type=int, default=-1, help="joblib n_jobs")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--output", type=Path, default=Path("reports/sweep_results.csv"))
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
    # IMPORTANT: preset="Custom" — otherwise atr_len/base_mult/sl_atr_mult/er_len/rsi_len
    # get overridden by the preset table and our sweep does nothing.
    base_cfg = SatsConfig(timeframe_minutes=args.timeframe_minutes, preset="Custom")

    print(f"Sweep: {len(grid)} combos × {len(df):,} bars  (objective: {args.objective})")
    results = grid_search(
        df, base_cfg, grid,
        objective=args.objective,
        n_jobs=args.jobs,
        verbose=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nFull results → {args.output}")
    print(f"\nTop {args.top} by {args.objective}:")
    cols = ["quality_strength", "quality_curve", "base_mult", "sl_atr_mult",
            "char_flip_min_age", "tp1_r", "tp2_r", "tp3_r",
            "n_trades", "win_rate", "avg_r", "total_r", "max_dd_r",
            "profit_factor", args.objective]
    print(results[cols].head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
