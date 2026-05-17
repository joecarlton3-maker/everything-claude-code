"""Config dataclass — mirrors Pine `input.*` calls 1:1.

Defaults match the Pine script exactly so a port-fidelity test can run
with no overrides. `resolve()` applies the preset and TP order-fix logic
from Pine section 3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

Preset = Literal["Auto", "Custom", "Scalping", "Default", "Swing", "Crypto 24/7"]
TpMode = Literal["Fixed", "Dynamic"]


@dataclass(frozen=True)
class SatsConfig:
    # Timeframe (minutes) — needed for Auto preset resolution.
    timeframe_minutes: int = 15

    # Main
    preset: Preset = "Auto"
    atr_len: int = 13
    base_mult: float = 2.0

    # Adaptive (legacy ER)
    use_adaptive: bool = True
    er_len: int = 20
    adapt_strength: float = 0.5
    atr_baseline_len: int = 100

    # Trend Quality Engine
    use_tqi: bool = True
    quality_strength: float = 0.4
    quality_curve: float = 1.5
    mult_smooth: bool = True

    use_asym_bands: bool = True
    asym_strength: float = 0.5

    use_eff_atr: bool = True

    use_char_flip: bool = True
    char_flip_min_age: int = 5
    char_flip_high: float = 0.55
    char_flip_low: float = 0.25

    tqi_weight_er: float = 0.35
    tqi_weight_vol: float = 0.20
    tqi_weight_struct: float = 0.25
    tqi_weight_mom: float = 0.20
    tqi_struct_len: int = 20
    tqi_mom_len: int = 10

    # Display-only filter params (kept for completeness)
    rsi_len: int = 14
    vol_len: int = 20

    # Risk
    sl_atr_mult: float = 1.5
    tp_mode: TpMode = "Fixed"
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    tp3_r: float = 3.0

    # Dynamic TP
    dyn_tp_tqi_weight: float = 0.6
    dyn_tp_vol_weight: float = 0.4
    dyn_tp_min_scale: float = 0.5
    dyn_tp_max_scale: float = 2.0
    dyn_tp_floor_r1: float = 0.5
    dyn_tp_ceil_r3: float = 8.0

    trade_max_age: int = 100

    # Smoothing constants (Pine: section 2)
    ewma_alpha: float = 0.2
    mult_smooth_alpha: float = 0.15

    # Regime thresholds (Pine: section 2)
    er_low_thresh: float = 0.25
    er_high_thresh: float = 0.50
    vol_low_thresh: float = 0.7
    vol_high_thresh: float = 1.3

    warmup_floor: int = 50


@dataclass(frozen=True)
class ResolvedConfig:
    """Config after preset application + TP ordering fix.

    Use `SatsConfig.resolve()` to produce one. Treats `cfg` fields as
    overridable when preset == 'Custom', otherwise overrides them.
    """
    raw: SatsConfig
    preset: str
    atr_len: int
    base_mult: float
    er_len: int
    rsi_len: int
    sl_atr_mult: float
    tp1_r: float
    tp2_r: float
    tp3_r: float

    @property
    def warmup_bars(self) -> int:
        c = self.raw
        return max(
            c.warmup_floor,
            max(
                self.atr_len,
                self.er_len,
                self.rsi_len,
                c.vol_len,
                c.tqi_mom_len,
                c.tqi_struct_len,
                c.atr_baseline_len,
            ) + 10,
        )


def _resolve_preset(preset: Preset, tf_minutes: int) -> str:
    if preset != "Auto":
        return preset
    if tf_minutes <= 5:
        return "Scalping"
    if tf_minutes <= 240:
        return "Default"
    return "Swing"


_PRESET_TABLE = {
    "Scalping":    dict(atr_len=10, base_mult=1.5, er_len=14, rsi_len=9,  sl_atr_mult=1.0),
    "Default":     dict(atr_len=14, base_mult=2.0, er_len=20, rsi_len=14, sl_atr_mult=1.5),
    "Swing":       dict(atr_len=21, base_mult=2.5, er_len=30, rsi_len=21, sl_atr_mult=2.0),
    "Crypto 24/7": dict(atr_len=14, base_mult=2.8, er_len=20, rsi_len=14, sl_atr_mult=2.5),
}


def resolve(cfg: SatsConfig) -> ResolvedConfig:
    resolved_preset = _resolve_preset(cfg.preset, cfg.timeframe_minutes)

    if resolved_preset == "Custom" or resolved_preset not in _PRESET_TABLE:
        atr_len = cfg.atr_len
        base_mult = cfg.base_mult
        er_len = cfg.er_len
        rsi_len = cfg.rsi_len
        sl_atr_mult = cfg.sl_atr_mult
    else:
        p = _PRESET_TABLE[resolved_preset]
        atr_len = p["atr_len"]
        base_mult = p["base_mult"]
        er_len = p["er_len"]
        rsi_len = p["rsi_len"]
        sl_atr_mult = p["sl_atr_mult"]

    # TP order fix — guarantee tp1 < tp2 < tp3 (Pine section 3.5)
    rs = sorted([cfg.tp1_r, cfg.tp2_r, cfg.tp3_r])
    tp1_r, tp2_r, tp3_r = rs[0], rs[1], rs[2]

    return ResolvedConfig(
        raw=cfg,
        preset=resolved_preset,
        atr_len=atr_len,
        base_mult=base_mult,
        er_len=er_len,
        rsi_len=rsi_len,
        sl_atr_mult=sl_atr_mult,
        tp1_r=tp1_r,
        tp2_r=tp2_r,
        tp3_r=tp3_r,
    )


# Convenience for tests
SatsConfig.resolve = resolve  # type: ignore[attr-defined]
SatsConfig.replace = replace  # type: ignore[attr-defined]
