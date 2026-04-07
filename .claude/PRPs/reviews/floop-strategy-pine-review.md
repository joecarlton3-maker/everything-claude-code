# Pine Script Review: FLOOP Strategy

**Reviewed**: 2026-04-07
**Author**: TheRealDrip2Rip (converted)
**Language**: Pine Script v5
**Decision**: REQUEST CHANGES

## Summary

The FLOOP Strategy is a well-structured multi-timeframe range-filter strategy with solid signal scoring. There are several performance issues that will cause slow chart loading and increased computation, plus a few visual/logic issues worth fixing.

## Findings

### HIGH

#### 1. ATR Rank Loop Recalculates 60 Bars Every Tick (Performance)

**Lines ~92-95** — The `for k = 0 to 59` loop manually counts how many of the last 60 bars have a lower ATR. This runs on every bar and is O(60) per bar.

**Fix**: Replace with `ta.percentrank(atr_val, 60)` which does the same thing natively and is optimized internally by Pine's runtime:

```pinescript
// BEFORE
atr_rank = 0.0
for k = 0 to 59
    if atr_val >= atr_val[k]
        atr_rank += 1
atr_rank := math.round(atr_rank / 60 * 100)

// AFTER
atr_rank = math.round(ta.percentrank(atr_val, 60))
```

This eliminates the only loop in the script and reduces per-bar cost significantly on large datasets.

#### 2. Five `request.security()` Calls Create Heavy Overhead (Performance)

The script makes **5 separate `request.security()` calls** (HTF, 5m, 15m, 60m, 240m), each invoking `f_range_filter()` which itself calls `ta.atr()`. Every `request.security()` call forces Pine to maintain a separate data series and re-evaluate the function on the higher timeframe.

**Mitigation**: This is architecturally intentional for MTF scoring, but consider:
- If deployed on a timeframe >= 60m, the 5m and 15m calls are requesting *lower* timeframes, which `request.security()` handles poorly (returns `na` or unreliable data). Add a guard or document the intended chart timeframe.
- The 240m call is redundant if the HTF input is already set to 240 — consider making MTF timeframes configurable or skipping when they match the chart/HTF.

#### 3. `f_range_filter()` Called 8 Times Total (Performance)

The range filter function is called:
- 1× on the current chart (line ~82)
- 5× via `request.security()` (HTF + 4 MTF)
- 2× for sensitivity cross-checks (sens=12, sens=16)

Each call includes `ta.atr()`. That's 8 ATR calculations plus 8 sets of var-state tracking. This is the main contributor to chart load time.

**Fix for sensitivity cross-checks**: Since `sC` (sens=12) and `sD` (sens=16) use the same ATR, cache the ATR and pass the range directly instead of recalculating:

```pinescript
// Reuse the already-calculated atr_val
rng_c = atr_val * i_atr_mult * (12 / 8.0)
rng_d = atr_val * i_atr_mult * (16 / 8.0)
// Then inline the filter logic for sC and sD using these ranges
```

This saves 2 redundant `ta.atr(14)` calculations per bar.

### MEDIUM

#### 4. Info Table Rebuilt Every Bar (Performance/Visual)

**Lines ~270-300** — The `table.new()` is `var` (created once), but all `table.cell()` calls run unconditionally on every bar. Tables are expensive in Pine Script.

**Fix**: Wrap the table updates in `barstate.islast` so they only render on the final bar:

```pinescript
if barstate.islast
    table.cell(info, 0, 0, ...)
    table.cell(info, 1, 0, ...)
    // ... all other table.cell calls
```

This is the single biggest visual performance improvement — tables are the #1 cause of slow rendering in Pine scripts.

#### 5. Trailing Stop Not Persisted Across `tp1_hit` Detection Bar (Logic)

When TP1 is detected (position shrinks), `trail_stop` is set to `math.max(entry_px, trail_stop)` for longs. But in the same bar's exit block, the code checks `tp1_hit` and applies RF trailing. Since `tp1_hit` was just set to `true` on this bar, the exit block immediately switches to Phase 2 trailing — this is correct but means the breakeven move and the first RF trail calculation happen simultaneously. If `rf_filt - atr_val * i_trail_pad` is below entry, `trail_stop` stays at breakeven (due to `math.max`), which is fine. No bug, but worth a comment for clarity.

#### 6. `daily_trades` Counter May Double-Count on Pyramiding Edge Cases (Logic)

The position-detection logic uses `strategy.position_size > 0 and strategy.position_size[1] <= 0` to detect new longs. With `pyramiding = 0` this is safe, but if pyramiding is ever enabled, re-entries without a flat bar would not increment the counter. Consider using `strategy.opentrades` for robustness.

#### 7. Band Fill Colors Asymmetric (Visual)

The upper band fill uses `c_band_b` (green) when bullish and `c_band_n` (neutral purple) otherwise. The lower band fill uses `c_band_r` (red) when bearish and `c_band_n` otherwise. This means during a bull trend, the *lower* band shows neutral purple — which is correct design, but during trend transitions both bands briefly flash purple, making it hard to see the flip. Consider a 1-bar transition color or increasing the band opacity from 85/95 to 75/85 for better visibility.

### LOW

#### 8. `score_vol` Uses Magic Numbers (Readability)

```pinescript
score_vol = (atr_rank < 80 ? 1 : 0) + (atr_norm < ta.percentile_nearest_rank(atr_norm, 60, 65) ? 1 : 0)
```

The `80` and `65` thresholds are hardcoded. For a strategy with 15+ configurable inputs, these should be inputs or at least named constants.

#### 9. Blocked Signal Shapes Could Be More Visible (Visual)

`plotshape` for blocked signals uses 70% transparency and `size.tiny`. On a busy chart with colored bars and bands, these are nearly invisible. Consider `size.small` and 50% transparency, or use `shape.triangledown`/`shape.triangleup` which are more distinct than `xcross`.

#### 10. `chop_range` Division-by-Zero Guard Returns 50.0 (Logic)

```pinescript
chop_index = chop_range > 0 ? ... : 50.0
```

A fallback of 50.0 is below the default threshold of 61.8, meaning the chop filter would **pass** when range is zero (a flat market). This is counterintuitive — a zero-range market is the definition of chop. Consider returning `100.0` instead to block signals in this edge case.

## Performance Impact Summary

| Issue | Est. Impact | Fix Effort |
|-------|------------|------------|
| Table on every bar (#4) | HIGH — main render bottleneck | 2 lines (wrap in `barstate.islast`) |
| ATR rank loop (#1) | MEDIUM — 60 iterations/bar | 1 line replacement |
| Redundant ATR in sens checks (#3) | LOW-MEDIUM — 2 extra ATR/bar | ~10 lines refactor |
| 5× request.security (#2) | MEDIUM — architectural | Document or gate |

## Recommended Priority

1. **Wrap table in `barstate.islast`** — instant chart speed improvement
2. **Replace ATR rank loop with `ta.percentrank()`** — cleaner and faster
3. **Fix chop_range fallback to 100.0** — logic correctness
4. **Cache ATR for sensitivity cross-checks** — moderate perf gain
5. **Increase blocked signal visibility** — usability

## Files Reviewed

| File | Type |
|------|------|
| FLOOP Strategy (inline) | Pine Script v5 Strategy |
