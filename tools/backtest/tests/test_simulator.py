"""Trade simulator tests — synthetic scenarios with hand-computed expected R.

The point of these tests is to lock down the R-multiple accounting so
parameter sweeps can trust the output. Each test constructs a tiny price
path designed to trigger a specific exit pathway.
"""
import numpy as np
import pandas as pd
import pytest

from sats.strategy.simulator import simulate


def _bars(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    if opens is None:
        opens = [closes[0]] + list(closes[:-1])
    if highs is None:
        highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
    if lows is None:
        lows = [min(o, c) - 0.1 for o, c in zip(opens, closes)]
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=idx,
    )


def _zeros(n):
    return pd.Series(np.zeros(n, dtype=bool))


def _const(n, v):
    return pd.Series(np.full(n, v, dtype=float))


def _run(df, *, flip_up=None, flip_down=None, atr=None, tp1=1.0, tp2=2.0, tp3=3.0,
         sl_mult=1.5, max_age=100, warmup=0):
    n = len(df)
    flip_up = flip_up if flip_up is not None else _zeros(n)
    flip_down = flip_down if flip_down is not None else _zeros(n)
    atr = atr if atr is not None else _const(n, 1.0)
    return simulate(
        df=df,
        flip_up=flip_up.reset_index(drop=True).set_axis(df.index),
        flip_down=flip_down.reset_index(drop=True).set_axis(df.index),
        atr_eff=atr.reset_index(drop=True).set_axis(df.index),
        last_pivot_high=pd.Series(np.nan, index=df.index),
        last_pivot_low=pd.Series(np.nan, index=df.index),
        tp1_r=_const(n, tp1).set_axis(df.index),
        tp2_r=_const(n, tp2).set_axis(df.index),
        tp3_r=_const(n, tp3).set_axis(df.index),
        sl_atr_mult=sl_mult,
        trade_max_age=max_age,
        warmup=warmup,
    )


class TestLongTrade:
    def test_long_hits_all_tps(self):
        # Entry at close=100, ATR=1.0, SL mult=1.5.
        # Pivot is NaN → sl_base = low[0] = 99.9
        # rawSl = 99.9 - 1.5 = 98.4;  minSl = 100 - 1.5 = 98.5;  sl = min = 98.4
        # risk = entry - sl = 1.6  →  TP1=101.6, TP2=103.2, TP3=104.8
        flip = _zeros(20)
        flip.iloc[0] = True
        closes = [100, 101, 102, 103, 104, 104, 104] + [104] * 13
        highs = [100.1, 101.7, 103.3, 104.9, 104.1, 104.1, 104.1] + [104.1] * 13
        lows = [99.9, 100.5, 101.5, 102.5, 103.5, 103.5, 103.5] + [103.5] * 13
        df = _bars(closes, highs=highs, lows=lows)
        res = _run(df, flip_up=flip)

        assert len(res.trades) == 1
        t = res.trades.iloc[0]
        assert t["direction"] == 1
        assert t["entry"] == pytest.approx(100.0)
        assert t["sl"] == pytest.approx(98.4)
        assert t["tp1"] == pytest.approx(101.6)
        assert t["tp3"] == pytest.approx(104.8)
        assert t["hit_tp1"] and t["hit_tp2"] and t["hit_tp3"]
        assert t["exit_reason"] == "tp3"
        # Realized R: (1 + 2 + 3) / 3 = 2.0  (R-multiples unchanged by SL price)
        assert t["realized_r"] == pytest.approx(2.0)

    def test_long_hits_tp1_then_sl(self):
        # Hit TP1 on bar 1, then SL on bar 2 → realized = 1/3 * 1 + 2/3 * (-1) = -1/3
        flip = _zeros(20)
        flip.iloc[0] = True
        closes = [100, 102, 97] + [97] * 17
        highs =  [100.1, 102.0, 100.5] + [97.1] * 17  # bar 1 high triggers TP1 at 101.5
        lows =   [99.9, 100.5, 97.0] + [96.9] * 17    # bar 2 low triggers SL at 98.5
        df = _bars(closes, highs=highs, lows=lows)
        res = _run(df, flip_up=flip)

        t = res.trades.iloc[0]
        assert t["hit_tp1"] and not t["hit_tp2"] and not t["hit_tp3"]
        assert t["exit_reason"] == "sl"
        # 1/3 at +1R, 2/3 at -1R = -0.333...
        assert t["realized_r"] == pytest.approx(1.0 / 3.0 - 2.0 / 3.0)

    def test_long_immediate_sl(self):
        flip = _zeros(20)
        flip.iloc[0] = True
        closes = [100, 95] + [95] * 18
        highs =  [100.1, 100.0] + [95.1] * 18
        lows =   [99.9, 95.0] + [94.9] * 18  # SL at 98.5 hit
        df = _bars(closes, highs=highs, lows=lows)
        res = _run(df, flip_up=flip)
        t = res.trades.iloc[0]
        assert t["exit_reason"] == "sl"
        assert t["realized_r"] == pytest.approx(-1.0)


class TestShortTrade:
    def test_short_hits_all_tps(self):
        # Entry at close=100, ATR=1.0, SL mult=1.5.
        # Pivot is NaN → sl_base = high[0] = 100.1
        # rawSl = 100.1 + 1.5 = 101.6;  minSl = 100 + 1.5 = 101.5;  sl = max = 101.6
        # risk = 1.6  →  TP1=98.4, TP2=96.8, TP3=95.2
        flip = _zeros(20)
        flip.iloc[0] = True
        closes = [100, 99, 98, 97, 95, 95, 95] + [95] * 13
        highs =  [100.1, 99.5, 98.5, 97.5, 96.0, 95.1, 95.1] + [95.1] * 13
        lows =   [99.9, 98.3, 96.7, 95.1, 94.9, 94.9, 94.9] + [94.9] * 13
        df = _bars(closes, highs=highs, lows=lows)
        res = _run(df, flip_down=flip)

        t = res.trades.iloc[0]
        assert t["direction"] == -1
        assert t["entry"] == pytest.approx(100.0)
        assert t["sl"] == pytest.approx(101.6)
        assert t["tp1"] == pytest.approx(98.4)
        assert t["tp3"] == pytest.approx(95.2)
        assert t["exit_reason"] == "tp3"
        assert t["realized_r"] == pytest.approx(2.0)


class TestTimeout:
    def test_timeout_with_no_tp_hit(self):
        flip = _zeros(50)
        flip.iloc[0] = True
        # Price drifts but doesn't touch any level → timeout
        closes = [100 + 0.01 * i for i in range(50)]  # tiny uptrend
        df = _bars(closes)
        res = _run(df, flip_up=flip, max_age=10)
        t = res.trades.iloc[0]
        assert t["exit_reason"] == "timeout"
        assert t["realized_r"] == 0.0
        assert t["bars_held"] == 10


class TestSignalFlip:
    def test_opposing_signal_closes_trade(self):
        flip_up = _zeros(20)
        flip_down = _zeros(20)
        flip_up.iloc[0] = True
        flip_down.iloc[5] = True  # short signal on bar 5 closes the long
        closes = [100] * 20
        df = _bars(closes)
        res = _run(df, flip_up=flip_up, flip_down=flip_down)

        assert len(res.trades) == 2
        first = res.trades.iloc[0]
        second = res.trades.iloc[1]
        assert first["direction"] == 1
        assert first["exit_reason"] == "signal_flip"
        assert second["direction"] == -1


class TestEquityCurve:
    def test_cumulative_r_matches_trade_sum(self):
        flip = _zeros(50)
        flip.iloc[0] = True
        flip.iloc[15] = True  # second long after first one closes (or flips)
        # Make first trade win, second timeout
        closes = (
            [100, 101, 102, 103, 104, 105, 105, 105, 105, 105]  # first hits TP3 by bar 3
            + [100] * 40
        )
        df = _bars(closes)
        res = _run(df, flip_up=flip, max_age=20)
        # Final equity should equal sum of realized R
        assert res.equity_r.iloc[-1] == pytest.approx(res.trades["realized_r"].sum())
