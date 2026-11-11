"""Tests for train.py — model architecture, sequence builder, data download."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from train import LOOKBACK_WINDOW, TRAIN_SPLIT, build_model, build_sequences, download_and_split


# ── build_sequences ───────────────────────────────────────────────────────────

class TestBuildSequences:
    """build_sequences turns a 1-D scaled array into (x, y) sliding-window pairs."""

    def _make_series(self, n: int) -> np.ndarray:
        return np.linspace(0, 1, n).reshape(-1, 1)

    # ---- shapes --------------------------------------------------------------

    def test_normal_shapes(self):
        series = self._make_series(200)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        assert x.shape == (100, LOOKBACK_WINDOW, 1)
        assert y.shape == (100,)

    def test_x_is_3d(self):
        x, _ = build_sequences(self._make_series(200), LOOKBACK_WINDOW)
        assert x.ndim == 3

    def test_y_is_1d(self):
        _, y = build_sequences(self._make_series(200), LOOKBACK_WINDOW)
        assert y.ndim == 1

    def test_x_and_y_same_length(self):
        series = self._make_series(300)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        assert len(x) == len(y)

    def test_sequence_count(self):
        n = 250
        x, y = build_sequences(self._make_series(n), LOOKBACK_WINDOW)
        assert len(x) == n - LOOKBACK_WINDOW

    # ---- edge: exact window + 1 row -----------------------------------------

    def test_exactly_window_plus_one_produces_one_sequence(self):
        series = self._make_series(LOOKBACK_WINDOW + 1)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        assert x.shape == (1, LOOKBACK_WINDOW, 1)
        assert y.shape == (1,)

    # ---- edge: fewer rows than window ----------------------------------------

    def test_fewer_rows_than_window_returns_empty(self):
        series = self._make_series(LOOKBACK_WINDOW - 1)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        assert len(x) == 0
        assert len(y) == 0

    def test_exactly_window_rows_returns_empty(self):
        series = self._make_series(LOOKBACK_WINDOW)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        assert len(x) == 0
        assert len(y) == 0

    # ---- correctness ---------------------------------------------------------

    def test_y_values_correspond_to_correct_positions(self):
        """y[i] must equal series[window + i], matching the sequence target."""
        series = self._make_series(150)
        x, y = build_sequences(series, LOOKBACK_WINDOW)
        for i, yi in enumerate(y):
            expected = series[LOOKBACK_WINDOW + i, 0]
            assert abs(yi - expected) < 1e-12

    def test_x_windows_are_contiguous(self):
        """Each window must be a contiguous slice of the input series."""
        series = self._make_series(150)
        x, _ = build_sequences(series, LOOKBACK_WINDOW)
        for i in range(len(x)):
            expected = series[i : i + LOOKBACK_WINDOW, 0]
            np.testing.assert_array_almost_equal(x[i, :, 0], expected)

    # ---- custom window sizes -------------------------------------------------

    @pytest.mark.parametrize("window", [1, 5, 50])
    def test_custom_window_sizes(self, window):
        n = window + 10
        series = self._make_series(n)
        x, y = build_sequences(series, window)
        assert len(x) == 10
        assert x.shape[1] == window


# ── build_model ───────────────────────────────────────────────────────────────

class TestBuildModel:
    """build_model returns a compiled Keras Sequential matching the spec."""

    @pytest.fixture(scope="class")
    def model(self):
        return build_model((LOOKBACK_WINDOW, 1))

    def test_model_is_not_none(self, model):
        assert model is not None

    def test_model_is_compiled(self, model):
        # A compiled model has an optimizer set
        assert model.optimizer is not None

    def test_input_shape(self, model):
        # first layer input: (batch, LOOKBACK_WINDOW, 1)
        expected = (None, LOOKBACK_WINDOW, 1)
        assert tuple(model.input_shape) == expected

    def test_output_shape_is_single_value(self, model):
        # final Dense(1) → (batch, 1)
        assert tuple(model.output_shape) == (None, 1)

    def test_layer_count(self, model):
        # 4 LSTM + 3 Dropout + 2 Dense = 9 layers (+ InputLayer = 10 in some TF versions)
        n = len(model.layers)
        assert 9 <= n <= 11

    def test_model_predicts_correct_shape(self, model):
        dummy_input = np.zeros((5, LOOKBACK_WINDOW, 1))
        preds = model.predict(dummy_input, verbose=0)
        assert preds.shape == (5, 1)

    def test_loss_is_mse(self, model):
        assert "mean_squared_error" in model.loss or model.loss == "mean_squared_error"

    def test_optimizer_is_adam(self, model):
        assert "adam" in type(model.optimizer).__name__.lower()


# ── download_and_split ────────────────────────────────────────────────────────

class TestDownloadAndSplit:
    """download_and_split fetches OHLCV data and returns scaled training sequences."""

    def _make_mock_df(self, n: int) -> pd.DataFrame:
        prices = np.linspace(100, 200, n)
        idx = pd.date_range("2010-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    # ---- happy path ----------------------------------------------------------

    def test_returns_correct_types(self):
        mock_df = self._make_mock_df(500)
        with patch("train.yf.download", return_value=mock_df):
            from datetime import date
            x, y, scaler = download_and_split(
                "FAKE", date(2020, 1, 1), date(2022, 1, 1)
            )
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        from sklearn.preprocessing import MinMaxScaler
        assert isinstance(scaler, MinMaxScaler)

    def test_x_shape_matches_window(self):
        mock_df = self._make_mock_df(500)
        with patch("train.yf.download", return_value=mock_df):
            from datetime import date
            x, _, _ = download_and_split("FAKE", date(2020, 1, 1), date(2022, 1, 1))
        assert x.shape[1] == LOOKBACK_WINDOW
        assert x.shape[2] == 1

    def test_x_and_y_same_length(self):
        mock_df = self._make_mock_df(500)
        with patch("train.yf.download", return_value=mock_df):
            from datetime import date
            x, y, _ = download_and_split("FAKE", date(2020, 1, 1), date(2022, 1, 1))
        assert len(x) == len(y)

    def test_training_values_in_unit_range(self):
        mock_df = self._make_mock_df(500)
        with patch("train.yf.download", return_value=mock_df):
            from datetime import date
            x, y, _ = download_and_split("FAKE", date(2020, 1, 1), date(2022, 1, 1))
        assert x.min() >= -1e-9
        assert x.max() <= 1 + 1e-9

    # ---- error cases ---------------------------------------------------------

    def test_empty_dataframe_calls_sys_exit(self):
        with patch("train.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(SystemExit) as exc_info:
                from datetime import date
                download_and_split("BADTICKER", date(2020, 1, 1), date(2022, 1, 1))
        assert exc_info.value.code == 1

    def test_training_too_small_calls_sys_exit(self):
        """50-row frame → training slice of 35 < LOOKBACK_WINDOW (100) → exit."""
        mock_df = self._make_mock_df(50)
        with patch("train.yf.download", return_value=mock_df):
            with pytest.raises(SystemExit) as exc_info:
                from datetime import date
                download_and_split("FAKE", date(2023, 1, 1), date(2024, 1, 1))
        assert exc_info.value.code == 1

    def test_single_row_calls_sys_exit(self):
        mock_df = self._make_mock_df(1)
        with patch("train.yf.download", return_value=mock_df):
            with pytest.raises(SystemExit):
                from datetime import date
                download_and_split("FAKE", date(2023, 1, 1), date(2024, 1, 1))

    # ---- ticker normalisation ------------------------------------------------

    def test_ticker_passed_to_yf_download(self):
        mock_df = self._make_mock_df(500)
        with patch("train.yf.download", return_value=mock_df) as mock_dl:
            from datetime import date
            download_and_split("AAPL", date(2020, 1, 1), date(2022, 1, 1))
        call_args = mock_dl.call_args
        assert call_args[0][0] == "AAPL" or call_args[1].get("ticker") == "AAPL" or mock_dl.call_args.args[0] == "AAPL"
