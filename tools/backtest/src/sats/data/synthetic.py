"""Synthetic OHLCV — deterministic random walk for tests / CI.

Not for backtesting (no real edge structure); only to let the pipeline
run end-to-end without a Databento key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(
    n_bars: int = 2000,
    start: str = "2024-01-01",
    freq: str = "15min",
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.5,
    starting_price: float = 4500.0,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with a GBM-ish close series.

    Each bar's OHLC is built so high ≥ max(open, close) and low ≤ min(...),
    which is what every indicator assumes.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, vol / 100.0, size=n_bars)
    close = starting_price * np.exp(np.cumsum(log_returns))

    # Build OHLC from close with small intra-bar wiggle
    wiggle = np.abs(rng.normal(0, vol / 200.0, size=(n_bars, 2))) * close[:, None]
    open_ = np.concatenate([[starting_price], close[:-1]])
    high = np.maximum(open_, close) + wiggle[:, 0]
    low = np.minimum(open_, close) - wiggle[:, 1]
    volume = rng.integers(100, 2000, size=n_bars).astype(float)

    idx = pd.date_range(start, periods=n_bars, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
