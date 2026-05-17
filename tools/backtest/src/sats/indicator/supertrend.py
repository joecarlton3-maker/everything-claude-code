"""Adaptive SuperTrend — Pine sections 6.2 and 6.3.

This is the stateful core. Pine semantics that MUST be preserved:

1. `prevTrend = stTrend[1]` — multipliers swap based on PRIOR bar's trend.
2. Active/passive bands:
       lowerMult = prevTrend == 1 ? activeMult : passiveMult
       upperMult = prevTrend == 1 ? passiveMult : activeMult
   (In an uptrend, the lower band is the trailing stop and gets the TIGHT
   active multiplier; the upper band — passive — gets the looser one.)
3. Ratchet:
       lowerBand = close[1] > lowerBand[1] ? max(raw, lowerBand[1]) : raw
       upperBand = close[1] < upperBand[1] ? min(raw, upperBand[1]) : raw
   The ratchet only fires while price stays on the trend side; on a break
   the band resets to the raw value (which permits the flip).
4. Multiplier smoothing (EMA on the multipliers themselves, not the bands)
   uses MULT_SMOOTH_ALPHA = 0.15. First bar seeds with raw value.
5. Character flip needs `trend_age = bar_index - trend_start_bar` and the
   PRIOR bar's TQI to detect quality collapse.

Implementation note: we ship a numba kernel for speed (parameter sweeps will
call this thousands of times). A pure-Python reference is kept in
`_supertrend_python` for cross-checking — tests verify both produce identical
output to ≤1e-9.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def deco(fn):
            return fn
        return deco if not args else args[0]


@dataclass
class SupertrendResult:
    st_line: pd.Series       # the active band (= lowerBand if trend==1 else upperBand)
    trend: pd.Series         # +1 / -1 / 0 (0 before warmup)
    lower_band: pd.Series
    upper_band: pd.Series
    active_mult: pd.Series   # final smoothed active multiplier (debug/dashboard)
    passive_mult: pd.Series
    flip_up: pd.Series       # bool: trend changed -1 → +1 this bar
    flip_down: pd.Series     # bool: trend changed +1 → -1 this bar


# ───────────────────────── numba kernel ─────────────────────────

@njit(cache=True)
def _supertrend_numba(
    source: np.ndarray,
    close: np.ndarray,
    atr_eff: np.ndarray,
    tqi: np.ndarray,
    er: np.ndarray,
    base_mult: float,
    use_adaptive: bool,
    adapt_strength: float,
    use_tqi: bool,
    quality_strength: float,
    quality_curve: float,
    use_asym: bool,
    asym_strength: float,
    use_smooth: bool,
    mult_smooth_alpha: float,
    use_char_flip: bool,
    char_high: float,
    char_low: float,
    char_min_age: int,
    warmup: int,
):
    n = source.shape[0]
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    active_mult_out = np.full(n, np.nan)
    passive_mult_out = np.full(n, np.nan)
    flip_up = np.zeros(n, dtype=np.bool_)
    flip_down = np.zeros(n, dtype=np.bool_)
    st_line = np.full(n, np.nan)

    active_mult_sm = np.nan
    passive_mult_sm = np.nan
    trend_start_bar = 0
    prev_tqi = 0.5

    for i in range(n):
        atr_i = atr_eff[i]
        if np.isnan(atr_i) or i < warmup:
            # Carry forward NaN bands; trend stays 0 until warm.
            if i > 0:
                trend[i] = trend[i - 1]
            prev_tqi = tqi[i] if not np.isnan(tqi[i]) else prev_tqi
            continue

        # Multiplier blend (Pine section 6.2)
        legacy = 1.0 + adapt_strength * (0.5 - er[i]) if use_adaptive else 1.0
        if use_tqi:
            qdev = (1.0 - tqi[i]) ** quality_curve
            tqi_mult = 1.0 - quality_strength + quality_strength * (0.6 + 0.8 * qdev)
        else:
            tqi_mult = 1.0
        sym_mult = base_mult * legacy * tqi_mult

        if use_tqi and use_asym:
            asym_tighten = 1.0 - asym_strength * tqi[i] * 0.3
            asym_widen = 1.0 + asym_strength * tqi[i] * 0.4
            active_raw = sym_mult * asym_tighten
            passive_raw = sym_mult * asym_widen
        else:
            active_raw = sym_mult
            passive_raw = sym_mult

        # EMA-smooth the multipliers
        if use_smooth:
            if np.isnan(active_mult_sm):
                active_mult_sm = active_raw
                passive_mult_sm = passive_raw
            else:
                active_mult_sm = active_mult_sm * (1.0 - mult_smooth_alpha) + active_raw * mult_smooth_alpha
                passive_mult_sm = passive_mult_sm * (1.0 - mult_smooth_alpha) + passive_raw * mult_smooth_alpha
        else:
            active_mult_sm = active_raw
            passive_mult_sm = passive_raw

        active_mult_out[i] = active_mult_sm
        passive_mult_out[i] = passive_mult_sm

        # Asymmetric band assignment based on PRIOR trend
        prev_trend = 1 if i == 0 else (trend[i - 1] if trend[i - 1] != 0 else 1)
        if prev_trend == 1:
            lower_mult = active_mult_sm
            upper_mult = passive_mult_sm
        else:
            lower_mult = passive_mult_sm
            upper_mult = active_mult_sm

        lower_raw = source[i] - lower_mult * atr_i
        upper_raw = source[i] + upper_mult * atr_i

        # Ratchet
        if i == 0 or np.isnan(lower[i - 1]):
            lower[i] = lower_raw
        else:
            if close[i - 1] > lower[i - 1]:
                lower[i] = max(lower_raw, lower[i - 1])
            else:
                lower[i] = lower_raw

        if i == 0 or np.isnan(upper[i - 1]):
            upper[i] = upper_raw
        else:
            if close[i - 1] < upper[i - 1]:
                upper[i] = min(upper_raw, upper[i - 1])
            else:
                upper[i] = upper_raw

        # Flip detection
        # Need the PREVIOUS bar's bands (already stored), not this bar's.
        if i == 0 or np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            price_flip_up = False
            price_flip_down = False
        else:
            price_flip_up = (prev_trend == -1) and (close[i] > upper[i - 1])
            price_flip_down = (prev_trend == 1) and (close[i] < lower[i - 1])

        trend_age = i - trend_start_bar
        if use_char_flip and use_tqi and not np.isnan(tqi[i]):
            cf_base = (prev_tqi > char_high) and (tqi[i] < char_low) and (trend_age >= char_min_age)
            cf_down = cf_base and (prev_trend == 1) and (close[i] < source[i])
            cf_up = cf_base and (prev_trend == -1) and (close[i] > source[i])
        else:
            cf_down = False
            cf_up = False

        final_flip_up = price_flip_up or cf_up
        final_flip_down = price_flip_down or cf_down

        if final_flip_up:
            new_trend = 1
        elif final_flip_down:
            new_trend = -1
        else:
            new_trend = prev_trend

        if new_trend != prev_trend or i == warmup:
            if new_trend != prev_trend:
                trend_start_bar = i

        trend[i] = new_trend
        st_line[i] = lower[i] if new_trend == 1 else upper[i]

        # Record flip flags
        if i > 0 and trend[i - 1] != 0:
            flip_up[i] = (new_trend == 1) and (trend[i - 1] == -1)
            flip_down[i] = (new_trend == -1) and (trend[i - 1] == 1)

        prev_tqi = tqi[i]

    return st_line, trend, lower, upper, active_mult_out, passive_mult_out, flip_up, flip_down


# ───────────────────────── pure-python reference ─────────────────────────

def _supertrend_python(
    source: np.ndarray,
    close: np.ndarray,
    atr_eff: np.ndarray,
    tqi: np.ndarray,
    er: np.ndarray,
    base_mult: float,
    use_adaptive: bool,
    adapt_strength: float,
    use_tqi: bool,
    quality_strength: float,
    quality_curve: float,
    use_asym: bool,
    asym_strength: float,
    use_smooth: bool,
    mult_smooth_alpha: float,
    use_char_flip: bool,
    char_high: float,
    char_low: float,
    char_min_age: int,
    warmup: int,
):
    """Reference implementation — identical math, no numba, for tests."""
    n = source.shape[0]
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    trend = np.zeros(n, dtype=np.int8)
    active_mult_out = np.full(n, np.nan)
    passive_mult_out = np.full(n, np.nan)
    flip_up = np.zeros(n, dtype=bool)
    flip_down = np.zeros(n, dtype=bool)
    st_line = np.full(n, np.nan)

    active_mult_sm = np.nan
    passive_mult_sm = np.nan
    trend_start_bar = 0
    prev_tqi = 0.5

    for i in range(n):
        atr_i = atr_eff[i]
        if np.isnan(atr_i) or i < warmup:
            if i > 0:
                trend[i] = trend[i - 1]
            if not np.isnan(tqi[i]):
                prev_tqi = tqi[i]
            continue

        legacy = 1.0 + adapt_strength * (0.5 - er[i]) if use_adaptive else 1.0
        if use_tqi:
            qdev = (1.0 - tqi[i]) ** quality_curve
            tqi_mult = 1.0 - quality_strength + quality_strength * (0.6 + 0.8 * qdev)
        else:
            tqi_mult = 1.0
        sym_mult = base_mult * legacy * tqi_mult

        if use_tqi and use_asym:
            active_raw = sym_mult * (1.0 - asym_strength * tqi[i] * 0.3)
            passive_raw = sym_mult * (1.0 + asym_strength * tqi[i] * 0.4)
        else:
            active_raw = sym_mult
            passive_raw = sym_mult

        if use_smooth:
            if np.isnan(active_mult_sm):
                active_mult_sm = active_raw
                passive_mult_sm = passive_raw
            else:
                active_mult_sm = active_mult_sm * (1.0 - mult_smooth_alpha) + active_raw * mult_smooth_alpha
                passive_mult_sm = passive_mult_sm * (1.0 - mult_smooth_alpha) + passive_raw * mult_smooth_alpha
        else:
            active_mult_sm = active_raw
            passive_mult_sm = passive_raw

        active_mult_out[i] = active_mult_sm
        passive_mult_out[i] = passive_mult_sm

        prev_trend = 1 if i == 0 else (trend[i - 1] if trend[i - 1] != 0 else 1)
        lower_mult = active_mult_sm if prev_trend == 1 else passive_mult_sm
        upper_mult = passive_mult_sm if prev_trend == 1 else active_mult_sm

        lower_raw = source[i] - lower_mult * atr_i
        upper_raw = source[i] + upper_mult * atr_i

        if i == 0 or np.isnan(lower[i - 1]):
            lower[i] = lower_raw
        else:
            lower[i] = max(lower_raw, lower[i - 1]) if close[i - 1] > lower[i - 1] else lower_raw

        if i == 0 or np.isnan(upper[i - 1]):
            upper[i] = upper_raw
        else:
            upper[i] = min(upper_raw, upper[i - 1]) if close[i - 1] < upper[i - 1] else upper_raw

        if i == 0 or np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            price_flip_up = False
            price_flip_down = False
        else:
            price_flip_up = (prev_trend == -1) and (close[i] > upper[i - 1])
            price_flip_down = (prev_trend == 1) and (close[i] < lower[i - 1])

        trend_age = i - trend_start_bar
        if use_char_flip and use_tqi and not np.isnan(tqi[i]):
            cf_base = (prev_tqi > char_high) and (tqi[i] < char_low) and (trend_age >= char_min_age)
            cf_down = cf_base and (prev_trend == 1) and (close[i] < source[i])
            cf_up = cf_base and (prev_trend == -1) and (close[i] > source[i])
        else:
            cf_down = False
            cf_up = False

        new_trend = 1 if (price_flip_up or cf_up) else (-1 if (price_flip_down or cf_down) else prev_trend)
        if new_trend != prev_trend:
            trend_start_bar = i

        trend[i] = new_trend
        st_line[i] = lower[i] if new_trend == 1 else upper[i]

        if i > 0 and trend[i - 1] != 0:
            flip_up[i] = (new_trend == 1) and (trend[i - 1] == -1)
            flip_down[i] = (new_trend == -1) and (trend[i - 1] == 1)

        prev_tqi = tqi[i]

    return st_line, trend, lower, upper, active_mult_out, passive_mult_out, flip_up, flip_down


# ───────────────────────── public API ─────────────────────────

def compute_supertrend(
    *,
    source: pd.Series,
    close: pd.Series,
    atr_eff: pd.Series,
    tqi: pd.Series,
    er: pd.Series,
    base_mult: float,
    use_adaptive: bool,
    adapt_strength: float,
    use_tqi: bool,
    quality_strength: float,
    quality_curve: float,
    use_asym: bool,
    asym_strength: float,
    use_smooth: bool,
    mult_smooth_alpha: float,
    use_char_flip: bool,
    char_high: float,
    char_low: float,
    char_min_age: int,
    warmup: int,
    backend: str = "numba",
) -> SupertrendResult:
    fn = _supertrend_numba if (backend == "numba" and HAS_NUMBA) else _supertrend_python
    st, trend, lower, upper, am, pm, fu, fd = fn(
        source.values.astype(np.float64),
        close.values.astype(np.float64),
        atr_eff.values.astype(np.float64),
        tqi.values.astype(np.float64),
        er.values.astype(np.float64),
        base_mult,
        use_adaptive,
        adapt_strength,
        use_tqi,
        quality_strength,
        quality_curve,
        use_asym,
        asym_strength,
        use_smooth,
        mult_smooth_alpha,
        use_char_flip,
        char_high,
        char_low,
        char_min_age,
        warmup,
    )
    idx = close.index
    return SupertrendResult(
        st_line=pd.Series(st, index=idx),
        trend=pd.Series(trend, index=idx).astype(int),
        lower_band=pd.Series(lower, index=idx),
        upper_band=pd.Series(upper, index=idx),
        active_mult=pd.Series(am, index=idx),
        passive_mult=pd.Series(pm, index=idx),
        flip_up=pd.Series(fu, index=idx),
        flip_down=pd.Series(fd, index=idx),
    )
