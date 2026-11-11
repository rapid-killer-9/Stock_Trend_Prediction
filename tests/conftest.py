"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


def _make_close(n: int, start_price: float = 100.0, seed: int = 42) -> pd.Series:
    """Return a deterministic synthetic closing-price series of length n."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, 0.01, size=n)
    prices = start_price * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="Close")


@pytest.fixture()
def close_1000():
    """1 000-day close series — well above any sequence threshold."""
    return _make_close(1000)


@pytest.fixture()
def close_150():
    """150-day close series — just enough for a few test sequences."""
    return _make_close(150)


@pytest.fixture()
def close_short():
    """50-day series — training slice (35 days) is smaller than LOOKBACK_WINDOW."""
    return _make_close(50)
