"""Dynamic TP scaling — Pine sections 5.3 and 6.35.

Computes, per-bar, the effective TP1/TP2/TP3 R-multiples for the signal
that would fire on this bar. Used by the trade simulator at entry time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .helpers import clamp, map_clamp


@dataclass
class DynamicTpResult:
    scale: pd.Series
    tp1_r: pd.Series
    tp2_r: pd.Series
    tp3_r: pd.Series


def compute_dynamic_tp(
    tqi: pd.Series,
    vol_ratio: pd.Series,
    *,
    base_tp1_r: float,
    base_tp2_r: float,
    base_tp3_r: float,
    tqi_weight: float,
    vol_weight: float,
    min_scale: float,
    max_scale: float,
    floor_r1: float,
    ceil_r3: float,
    enabled: bool,
) -> DynamicTpResult:
    if not enabled:
        ones = pd.Series(1.0, index=tqi.index)
        return DynamicTpResult(
            scale=ones,
            tp1_r=pd.Series(base_tp1_r, index=tqi.index),
            tp2_r=pd.Series(base_tp2_r, index=tqi.index),
            tp3_r=pd.Series(base_tp3_r, index=tqi.index),
        )

    tqi_comp = clamp(tqi, 0.0, 1.0)
    vol_comp = clamp(map_clamp(vol_ratio, 0.5, 2.0, 0.0, 1.0), 0.0, 1.0)
    wsum = tqi_weight + vol_weight
    wden = wsum if wsum > 0 else 1.0
    raw = (tqi_comp * tqi_weight + vol_comp * vol_weight) / wden
    scale = min_scale + raw * (max_scale - min_scale)

    # Per-level proportional floors (Pine section 6.35):
    #   tp1_floor = floor_r1
    #   tp2_floor = floor_r1 * (base_tp2 / base_tp1)
    #   tp3_floor = floor_r1 * (base_tp3 / base_tp1)
    base_tp1_safe = max(base_tp1_r, 0.01)
    tp1_floor = floor_r1
    tp2_floor = floor_r1 * (base_tp2_r / base_tp1_safe)
    tp3_floor = floor_r1 * (base_tp3_r / base_tp1_safe)

    tp1 = (base_tp1_r * scale).clip(lower=tp1_floor, upper=ceil_r3)
    tp2 = (base_tp2_r * scale).clip(lower=tp2_floor, upper=ceil_r3)
    tp3 = (base_tp3_r * scale).clip(lower=tp3_floor, upper=ceil_r3)

    # Re-sort so tp1 <= tp2 <= tp3 (Pine: re-sort after scaling)
    stacked = np.sort(np.column_stack([tp1.values, tp2.values, tp3.values]), axis=1)
    tp1_sorted = pd.Series(stacked[:, 0], index=tqi.index)
    tp2_sorted = pd.Series(stacked[:, 1], index=tqi.index)
    tp3_sorted = pd.Series(stacked[:, 2], index=tqi.index)

    return DynamicTpResult(
        scale=scale,
        tp1_r=tp1_sorted,
        tp2_r=tp2_sorted,
        tp3_r=tp3_sorted,
    )
