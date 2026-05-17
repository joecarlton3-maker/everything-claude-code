"""Trend Quality Index — Pine section 6.1.

Continuous 0..1 score, weighted blend of 4 factors:
    Efficiency (default 35%)
    Volatility/Volume (default 20%)
    Structure (default 25%)
    Momentum persistence (default 20%)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .helpers import clamp, efficiency_ratio, map_clamp, volume_zscore


@dataclass
class TqiComponents:
    er: pd.Series         # tqiEr (== clamped ER)
    vol: pd.Series        # tqiVol
    struct: pd.Series     # tqiStruct
    mom: pd.Series        # tqiMom
    tqi: pd.Series        # final blended value


def compute_tqi(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series | None,
    er_series: pd.Series,
    vol_ratio: pd.Series,
    *,
    use_tqi: bool,
    weight_er: float,
    weight_vol: float,
    weight_struct: float,
    weight_mom: float,
    struct_len: int,
    mom_len: int,
    vol_z_len: int,
) -> TqiComponents:
    """Compute TQI and its 4 components.

    `er_series` and `vol_ratio` are passed in (not recomputed) so callers
    can reuse them for the SuperTrend band logic — keeps a single source
    of truth.
    """
    # ── Efficiency component ──
    tqi_er = clamp(er_series, 0.0, 1.0)

    # ── Volume / volatility-regime component ──
    has_volume = volume is not None and (volume.fillna(0) > 0).any()
    if has_volume:
        z = volume_zscore(volume, vol_z_len)
        tqi_vol = clamp(map_clamp(z, -1.0, 2.0, 0.0, 1.0), 0.0, 1.0)
    else:
        tqi_vol = clamp(map_clamp(vol_ratio, 0.6, 1.8, 0.0, 1.0), 0.0, 1.0)

    # ── Structure component ──
    struct_hi = high.rolling(struct_len).max()
    struct_lo = low.rolling(struct_len).min()
    struct_range = struct_hi - struct_lo
    price_pos = (close - struct_lo) / struct_range.replace(0, np.nan)
    price_pos = price_pos.fillna(0.5)
    tqi_struct = clamp((price_pos - 0.5).abs() * 2.0, 0.0, 1.0)

    # ── Momentum persistence ──
    # alignedBars = count of bars whose 1-bar delta has same sign as the N-bar net change.
    window_change = close - close.shift(mom_len)
    bar_change = close.diff()
    aligned = ((window_change > 0) & (bar_change > 0)) | ((window_change < 0) & (bar_change < 0))
    aligned_count = aligned.astype(float).rolling(mom_len).sum()
    tqi_mom = (aligned_count / float(mom_len)).fillna(0.0)

    if use_tqi:
        wsum = weight_er + weight_vol + weight_struct + weight_mom
        wden = wsum if wsum > 0 else 1.0
        tqi_raw = (
            tqi_er * weight_er
            + tqi_vol * weight_vol
            + tqi_struct * weight_struct
            + tqi_mom * weight_mom
        ) / wden
        tqi = clamp(tqi_raw, 0.0, 1.0)
    else:
        tqi = pd.Series(0.5, index=close.index)

    return TqiComponents(
        er=tqi_er.astype(float),
        vol=tqi_vol.astype(float),
        struct=tqi_struct.astype(float),
        mom=tqi_mom.astype(float),
        tqi=tqi.astype(float),
    )
