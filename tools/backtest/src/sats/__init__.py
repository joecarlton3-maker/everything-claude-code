"""SATS — Self-Aware Trend System backtest harness.

Python port of the Pine v6 indicator for parameter tuning and walk-forward
validation. Module map:

    sats.config              dataclass mirroring Pine inputs (incl. preset resolution)
    sats.indicator.helpers   atomic math: clamp/mapClamp/ER/ATR/RSI
    sats.indicator.tqi       Trend Quality Index (4-factor blend)
    sats.indicator.supertrend  adaptive SuperTrend w/ asymmetric bands + char-flip
    sats.indicator.dynamic_tp  dynamic TP R-multiple scaling
    sats.data.databento_loader  CME futures fetch + cache
    sats.strategy.simulator  bar-by-bar trade execution (numba)
    sats.backtest.runner     vectorbt/quantstats wiring + reports
"""
__version__ = "0.1.0"
