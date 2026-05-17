"""Pytest fixtures + sys.path bootstrap."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from sats.data.synthetic import synthetic_ohlcv  # noqa: E402


@pytest.fixture
def df_small() -> pd.DataFrame:
    """200-bar synthetic OHLCV for fast tests."""
    return synthetic_ohlcv(n_bars=200, seed=7)


@pytest.fixture
def df_medium() -> pd.DataFrame:
    """1000-bar synthetic OHLCV — enough for warmup + interesting dynamics."""
    return synthetic_ohlcv(n_bars=1000, seed=11)
