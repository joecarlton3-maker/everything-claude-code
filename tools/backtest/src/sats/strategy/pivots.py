"""Pivot detection — port of Pine `ta.pivothigh` / `ta.pivotlow`.

Pine semantics:
    `ta.pivothigh(high, left, right)` at bar i returns high[right] iff
    high[right] is STRICTLY greater than high[i-left-right .. i-right-1]
    AND high[right] is STRICTLY greater than high[i-right+1 .. i].
    Otherwise returns na.

So pivots are detected with `right` bars of lag — once `right` bars have
passed without exceeding the candidate, we confirm it as a pivot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
    """Returns the pivot value at bar i (offset by `right` from the pivot bar)."""
    h = np.asarray(high, dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        c = h[center]
        is_pivot = True
        # Strict greater than all `left` bars to the left
        for j in range(1, left + 1):
            if h[center - j] >= c:
                is_pivot = False
                break
        if is_pivot:
            # Strict greater than all `right` bars to the right
            for j in range(1, right + 1):
                if h[center + j] >= c:
                    is_pivot = False
                    break
        if is_pivot:
            out[i] = c
    return pd.Series(out, index=high.index)


def pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
    """Strict pivot low — mirror of `pivot_high`."""
    l = np.asarray(low, dtype=float)
    n = len(l)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        c = l[center]
        is_pivot = True
        for j in range(1, left + 1):
            if l[center - j] <= c:
                is_pivot = False
                break
        if is_pivot:
            for j in range(1, right + 1):
                if l[center + j] <= c:
                    is_pivot = False
                    break
        if is_pivot:
            out[i] = c
    return pd.Series(out, index=low.index)


def last_pivot(pivots: pd.Series) -> pd.Series:
    """Forward-fill the most recent pivot value (mirrors Pine `var float last`)."""
    return pivots.ffill()
