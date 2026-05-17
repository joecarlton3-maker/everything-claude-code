"""Atomic math primitives — direct ports of Pine functions in section 5.

These are the only functions that should hand-translate Pine semantics
(safe division, NaN handling, Wilder smoothing). Everything else builds
on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def safe_div(num: pd.Series | float, den: pd.Series | float, fallback: float = 0.0) -> pd.Series | float:
    """Port of `safeDiv` — returns fallback when denominator is 0 or NaN."""
    if isinstance(den, pd.Series) or isinstance(num, pd.Series):
        n = num if isinstance(num, pd.Series) else pd.Series([num])
        d = den if isinstance(den, pd.Series) else pd.Series([den])
        out = n / d.replace(0, np.nan)
        return out.fillna(fallback)
    if den == 0 or np.isnan(den) or np.isnan(num):
        return fallback
    return num / den


def clamp(v, lo: float, hi: float):
    """Element-wise clamp."""
    if isinstance(v, pd.Series):
        return v.clip(lower=lo, upper=hi)
    return max(lo, min(hi, v))


def map_clamp(v, in_lo: float, in_hi: float, out_lo: float, out_hi: float):
    """Port of `mapClamp` — linear remap with clamping at the input boundaries.

    NOTE: Pine clamps the *normalized* t to [0,1] before applying it to the
    output range. We do the same to preserve the boundary behavior even when
    out_hi < out_lo (inverted mapping is handled by `map_clamp_inv`).
    """
    rng = in_hi - in_lo
    if rng == 0:
        t = 0.0 if not isinstance(v, pd.Series) else pd.Series(0.0, index=v.index)
    else:
        t = (v - in_lo) / rng
        t = clamp(t, 0.0, 1.0)
    return out_lo + t * (out_hi - out_lo)


def map_clamp_inv(v, in_lo: float, in_hi: float, out_high: float, out_low: float):
    """Port of `mapClampInv` — inverse linear remap."""
    rng = in_hi - in_lo
    if rng == 0:
        t = 0.0 if not isinstance(v, pd.Series) else pd.Series(0.0, index=v.index)
    else:
        t = (v - in_lo) / rng
        t = clamp(t, 0.0, 1.0)
    return out_high - t * (out_high - out_low)


def efficiency_ratio(src: pd.Series, length: int) -> pd.Series:
    """Port of `calcEfficiencyRatio`.

        change      = |src - src[len]|
        volatility  = Σ|src - src[1]| over len bars
        ER          = change / volatility   (safe_div)
    """
    change = (src - src.shift(length)).abs()
    vol = src.diff().abs().rolling(length).sum()
    er = change / vol.replace(0, np.nan)
    return er.fillna(0.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def rma(s: pd.Series, length: int) -> pd.Series:
    """Wilder's moving average — Pine's `ta.rma` (and the basis for ta.atr/ta.rsi).

    Pine seeds RMA with an SMA of the first `length` values, then applies
    α = 1/length recursively.
    """
    s = s.copy()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < length:
        return out
    # Seed: SMA of first `length` non-NaN values.
    first_valid = s.first_valid_index()
    if first_valid is None:
        return out
    start_loc = s.index.get_loc(first_valid)
    seed_end = start_loc + length
    if seed_end > len(s):
        return out
    seed = s.iloc[start_loc:seed_end].mean()
    alpha = 1.0 / length
    vals = np.asarray(s, dtype=float)
    out_vals = np.full(len(s), np.nan, dtype=float)
    out_vals[seed_end - 1] = seed
    for i in range(seed_end, len(s)):
        v = vals[i]
        prev = out_vals[i - 1]
        if np.isnan(v):
            out_vals[i] = prev
        else:
            out_vals[i] = prev + alpha * (v - prev)
    return pd.Series(out_vals, index=s.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Port of `ta.atr(length)` — RMA of true range."""
    return rma(true_range(high, low, close), length)


def rsi(close: pd.Series, length: int) -> pd.Series:
    """Port of `ta.rsi(close, length)`."""
    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = (-delta.clip(upper=0)).fillna(0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100.0)
    return out


def stdev_pop(s: pd.Series, length: int) -> pd.Series:
    """Population stdev — matches Pine's `ta.stdev`."""
    return s.rolling(length).std(ddof=0)


def volume_zscore(volume: pd.Series, length: int) -> pd.Series:
    """Port of `calcVolumeZ` — (vol - SMA) / population stdev."""
    mean = volume.rolling(length).mean()
    std = stdev_pop(volume, length)
    z = (volume - mean) / std.replace(0, np.nan)
    return z.fillna(0.0)
