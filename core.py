"""Pure business-logic functions shared by app.py and the test suite."""

from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

TRAIN_SPLIT = 0.70
LOOKBACK_WINDOW = 100
MODEL_PATH = "Keras_Model.h5"

DATE_RANGE_OPTIONS = ["10-Year", "1-Year", "2-Year", "3-Year", "5-Year", "20-Year"]

_DELTA_MAP: dict[str, relativedelta] = {
    "1-Year": relativedelta(years=-1),
    "2-Year": relativedelta(years=-2),
    "3-Year": relativedelta(years=-3),
    "5-Year": relativedelta(years=-5),
    "10-Year": relativedelta(years=-10),
    "20-Year": relativedelta(years=-20),
}


def start_date_for_range(selected: str) -> date:
    delta = _DELTA_MAP.get(selected)
    if delta is None:
        return date.today()
    return date.today() + delta


def prepare_sequences(
    close: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split close prices into scaled test sequences and return the scale factor.

    The scaler is fitted exclusively on the training slice so that no test-period
    price information leaks into the normalisation step.
    """
    split_idx = int(len(close) * TRAIN_SPLIT)
    training = pd.DataFrame(close[:split_idx])
    testing = pd.DataFrame(close[split_idx:])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training)  # fit on training slice only — no data leakage

    seed_window = training.tail(LOOKBACK_WINDOW)
    combined = pd.concat([seed_window, testing], ignore_index=True)
    scaled = scaler.transform(combined)  # transform only

    x_test, y_test = [], []
    for i in range(LOOKBACK_WINDOW, scaled.shape[0]):
        x_test.append(scaled[i - LOOKBACK_WINDOW : i])
        y_test.append(scaled[i, 0])

    scale_factor = 1.0 / scaler.scale_[0]
    return np.array(x_test), np.array(y_test), scale_factor
