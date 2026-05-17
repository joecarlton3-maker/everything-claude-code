"""Databento loader for CME futures continuous front-month bars.

Fetches 1-minute OHLCV via Databento's historical API, aggregates to the
requested timeframe, and caches to parquet to avoid re-downloading on every
sweep.

Cost note: at ~$0.05 / symbol-month for ohlcv-1m on the GLBX.MDP3 dataset,
5y of ES is roughly $3-5. Cache aggressively.

Requires DATABENTO_API_KEY env var (or pass `api_key=` explicitly).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False
    db = None  # type: ignore[assignment]


DEFAULT_CACHE = Path(os.environ.get("SATS_CACHE_DIR", "data_cache"))
DEFAULT_DATASET = "GLBX.MDP3"


def _cache_key(symbol: str, start: str, end: str, schema: str) -> str:
    """Stable hash for cache filenames so we don't collide on overlapping ranges."""
    h = hashlib.sha1(f"{symbol}|{start}|{end}|{schema}".encode()).hexdigest()[:12]
    return f"{symbol}_{start}_{end}_{schema}_{h}"


def load_from_databento(
    symbol: str,
    start: str,
    end: str,
    *,
    schema: str = "ohlcv-1m",
    dataset: str = DEFAULT_DATASET,
    stype_in: str = "continuous",
    api_key: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch raw OHLCV from Databento with on-disk parquet cache.

    Returns DataFrame indexed by UTC timestamp with columns:
    open, high, low, close, volume.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_cache_key(symbol, start, end, schema)}.parquet"

    if cache_file.exists() and not force_refresh:
        return pd.read_parquet(cache_file)

    if not HAS_DATABENTO:
        raise ImportError(
            "databento package not installed. Run `pip install databento`."
        )
    key = api_key or os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise ValueError(
            "DATABENTO_API_KEY not set. Either export it or pass api_key= explicitly."
        )

    client = db.Historical(key)
    data = client.timeseries.get_range(
        dataset=dataset,
        schema=schema,
        symbols=[symbol],
        stype_in=stype_in,
        start=start,
        end=end,
    )
    df = data.to_df()
    # Databento returns columns like ts_event/ts_recv (already index in to_df).
    # Standardize to lowercase OHLCV.
    df = df.rename(columns={c: c.lower() for c in df.columns})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.to_parquet(cache_file)
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1m OHLCV to a coarser bar (e.g. '15min', '1h')."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule, label="left", closed="left").agg(cols)
    return out.dropna(subset=["open", "high", "low", "close"])


def load_es_15m(start: str, end: str, **kw) -> pd.DataFrame:
    """ES continuous front-month, aggregated to 15-minute bars."""
    raw = load_from_databento("ES.c.0", start, end, schema="ohlcv-1m", **kw)
    return resample_ohlcv(raw, "15min")


def load_nq_15m(start: str, end: str, **kw) -> pd.DataFrame:
    """NQ continuous front-month, aggregated to 15-minute bars."""
    raw = load_from_databento("NQ.c.0", start, end, schema="ohlcv-1m", **kw)
    return resample_ohlcv(raw, "15min")
