"""End-to-end indicator pipeline — orchestrates Pine sections 6.0 → 6.35.

Inputs: OHLCV DataFrame + SatsConfig.
Outputs: a `BarState` bundle with everything a downstream simulator needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SatsConfig, resolve
from .dynamic_tp import DynamicTpResult, compute_dynamic_tp
from .helpers import atr as atr_fn
from .helpers import efficiency_ratio
from .supertrend import SupertrendResult, compute_supertrend
from .tqi import TqiComponents, compute_tqi


@dataclass
class BarState:
    """Everything a trade simulator needs per bar."""
    index: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    atr_raw: pd.Series       # ta.atr(len), no efficiency weighting
    atr_eff: pd.Series       # used for band/SL (== atr_raw when use_eff_atr=False)
    er: pd.Series
    vol_ratio: pd.Series
    tqi: TqiComponents
    supertrend: SupertrendResult
    dynamic_tp: DynamicTpResult
    warmup_bars: int


def compute_bar_state(df: pd.DataFrame, cfg: SatsConfig) -> BarState:
    """Run the full indicator pipeline on an OHLCV DataFrame.

    `df` must have columns: open, high, low, close. `volume` is optional but
    recommended for ES/NQ.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    rc = resolve(cfg)
    src = df["close"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"] if "volume" in df.columns else None

    # ── Base calcs (Pine section 6) ──
    atr_raw = atr_fn(high, low, close, rc.atr_len).fillna(0.0)
    atr_baseline = atr_raw.rolling(cfg.atr_baseline_len).mean()
    vol_ratio = (atr_raw / atr_baseline.replace(0, np.nan)).fillna(1.0)
    er = efficiency_ratio(close, rc.er_len)

    atr_eff = atr_raw * (0.5 + 0.5 * er) if cfg.use_eff_atr else atr_raw

    # ── TQI ──
    tqi_components = compute_tqi(
        close=close,
        high=high,
        low=low,
        volume=volume,
        er_series=er,
        vol_ratio=vol_ratio,
        use_tqi=cfg.use_tqi,
        weight_er=cfg.tqi_weight_er,
        weight_vol=cfg.tqi_weight_vol,
        weight_struct=cfg.tqi_weight_struct,
        weight_mom=cfg.tqi_weight_mom,
        struct_len=cfg.tqi_struct_len,
        mom_len=cfg.tqi_mom_len,
        vol_z_len=cfg.vol_len,
    )

    # ── Adaptive SuperTrend ──
    st = compute_supertrend(
        source=src,
        close=close,
        atr_eff=atr_eff,
        tqi=tqi_components.tqi,
        er=er,
        base_mult=rc.base_mult,
        use_adaptive=cfg.use_adaptive,
        adapt_strength=cfg.adapt_strength,
        use_tqi=cfg.use_tqi,
        quality_strength=cfg.quality_strength,
        quality_curve=cfg.quality_curve,
        use_asym=cfg.use_asym_bands,
        asym_strength=cfg.asym_strength,
        use_smooth=cfg.mult_smooth,
        mult_smooth_alpha=cfg.mult_smooth_alpha,
        use_char_flip=cfg.use_char_flip,
        char_high=cfg.char_flip_high,
        char_low=cfg.char_flip_low,
        char_min_age=cfg.char_flip_min_age,
        warmup=rc.warmup_bars,
    )

    # ── Dynamic TP ──
    dtp = compute_dynamic_tp(
        tqi=tqi_components.tqi,
        vol_ratio=vol_ratio,
        base_tp1_r=rc.tp1_r,
        base_tp2_r=rc.tp2_r,
        base_tp3_r=rc.tp3_r,
        tqi_weight=cfg.dyn_tp_tqi_weight,
        vol_weight=cfg.dyn_tp_vol_weight,
        min_scale=cfg.dyn_tp_min_scale,
        max_scale=cfg.dyn_tp_max_scale,
        floor_r1=cfg.dyn_tp_floor_r1,
        ceil_r3=cfg.dyn_tp_ceil_r3,
        enabled=cfg.tp_mode == "Dynamic",
    )

    return BarState(
        index=df.index,
        open=df["open"],
        high=high,
        low=low,
        close=close,
        atr_raw=atr_raw,
        atr_eff=atr_eff,
        er=er,
        vol_ratio=vol_ratio,
        tqi=tqi_components,
        supertrend=st,
        dynamic_tp=dtp,
        warmup_bars=rc.warmup_bars,
    )
