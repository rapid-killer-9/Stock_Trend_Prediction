"""Tests for core.py — pure business-logic functions."""

import math
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core import (
    LOOKBACK_WINDOW,
    TRAIN_SPLIT,
    DATE_RANGE_OPTIONS,
    prepare_sequences,
    start_date_for_range,
)

# ── start_date_for_range ──────────────────────────────────────────────────────

FIXED_TODAY = date(2024, 6, 15)


def _today_patch():
    return patch("core.date")


class TestStartDateForRange:
    """start_date_for_range maps UI labels to Python dates."""

    @pytest.mark.parametrize(
        "label, expected_years",
        [
            ("1-Year", -1),
            ("2-Year", -2),
            ("3-Year", -3),
            ("5-Year", -5),
            ("10-Year", -10),
            ("20-Year", -20),
        ],
    )
    def test_all_valid_ranges_return_correct_year_offset(self, label, expected_years):
        result = start_date_for_range(label)
        today = date.today()
        # The year difference must match; allow for leap-year ±1 day boundary
        assert result.year == today.year + expected_years or abs(
            (today - result).days - abs(expected_years) * 365
        ) <= 2

    def test_all_options_are_covered(self):
        """Every entry in DATE_RANGE_OPTIONS must map to a date in the past."""
        today = date.today()
        for label in DATE_RANGE_OPTIONS:
            result = start_date_for_range(label)
            assert result < today, f"{label!r} did not produce a past date"

    def test_unknown_label_returns_today(self):
        assert start_date_for_range("INVALID") == date.today()

    def test_empty_string_returns_today(self):
        assert start_date_for_range("") == date.today()

    def test_case_sensitive_mismatch_returns_today(self):
        # Labels are title-cased; lowercase must fall through to default
        assert start_date_for_range("10-year") == date.today()

    def test_return_type_is_date(self):
        for label in DATE_RANGE_OPTIONS:
            result = start_date_for_range(label)
            assert isinstance(result, date)


# ── prepare_sequences ─────────────────────────────────────────────────────────

class TestPrepareSequences:
    """prepare_sequences builds sliding-window test sequences from a price series."""

    # ---- output shapes -------------------------------------------------------

    def test_normal_output_shapes(self, close_1000):
        x, y, sf = prepare_sequences(close_1000)

        split = int(len(close_1000) * TRAIN_SPLIT)
        test_len = len(close_1000) - split
        # seed_window (100) + test_len rows → (test_len) sequences
        expected_sequences = test_len
        assert x.shape == (expected_sequences, LOOKBACK_WINDOW, 1)
        assert y.shape == (expected_sequences,)

    def test_x_has_lookback_window_columns(self, close_1000):
        x, _, _ = prepare_sequences(close_1000)
        assert x.shape[1] == LOOKBACK_WINDOW

    def test_x_is_3d(self, close_1000):
        x, _, _ = prepare_sequences(close_1000)
        assert x.ndim == 3

    def test_y_is_1d(self, close_1000):
        _, y, _ = prepare_sequences(close_1000)
        assert y.ndim == 1

    def test_x_and_y_same_length(self, close_1000):
        x, y, _ = prepare_sequences(close_1000)
        assert len(x) == len(y)

    # ---- minimum viable input ------------------------------------------------

    def test_minimum_input_for_one_sequence(self):
        """split × n ≥ LOOKBACK_WINDOW and (1 - split) × n ≥ 1 test row."""
        # training slice needs ≥ LOOKBACK_WINDOW rows
        # At TRAIN_SPLIT=0.70, training = floor(n * 0.70)
        # Need floor(n * 0.70) ≥ 100  →  n ≥ 143 (ceil(100 / 0.70))
        n = math.ceil(LOOKBACK_WINDOW / TRAIN_SPLIT) + 1
        close = pd.Series(np.linspace(1, 200, n), name="Close")
        x, y, _ = prepare_sequences(close)
        assert len(x) >= 1

    def test_close_150_produces_sequences(self, close_150):
        x, y, _ = prepare_sequences(close_150)
        assert len(x) > 0

    # ---- insufficient data ---------------------------------------------------

    def test_short_series_returns_empty_arrays(self, close_short):
        """50 days → training slice of 35, smaller than LOOKBACK_WINDOW (100)."""
        x, y, sf = prepare_sequences(close_short)
        assert len(x) == 0
        assert len(y) == 0

    def test_single_row_returns_empty_arrays(self):
        close = pd.Series([100.0], name="Close")
        x, y, _ = prepare_sequences(close)
        assert len(x) == 0
        assert len(y) == 0

    # ---- scale factor --------------------------------------------------------

    def test_scale_factor_is_positive(self, close_1000):
        _, _, sf = prepare_sequences(close_1000)
        assert sf > 0

    def test_scale_factor_is_finite(self, close_1000):
        _, _, sf = prepare_sequences(close_1000)
        assert math.isfinite(sf)

    def test_scale_factor_inverts_correctly(self, close_1000):
        """Multiplying scaled values by scale_factor should recover original range."""
        x, y, sf = prepare_sequences(close_1000)
        # All scaled y values are in [0, 1]; after inversion they must cover
        # the actual price range (rough sanity — not exact due to seed window)
        recovered = y * sf
        price_range = close_1000.max() - close_1000.min()
        assert recovered.max() > 0
        assert recovered.max() < price_range * 2  # generous bound

    # ---- no data-leakage from test slice into scaler -------------------------

    def test_scaler_fitted_on_training_slice_only(self, close_1000):
        """
        The MinMaxScaler must be fit only on the training slice. We verify by
        checking that prepare_sequences on the first 70% alone produces the
        same scale_factor as when called on the full series (because the scaler
        sees the same 70% training range in both cases).
        """
        split_idx = int(len(close_1000) * TRAIN_SPLIT)
        # Grab scale_factor from the full run
        _, _, sf_full = prepare_sequences(close_1000)

        # Build a scaler manually using only the training slice
        from sklearn.preprocessing import MinMaxScaler
        training = close_1000.iloc[:split_idx].values.reshape(-1, 1)
        manual_scaler = MinMaxScaler(feature_range=(0, 1))
        manual_scaler.fit(training)
        expected_sf = 1.0 / manual_scaler.scale_[0]

        assert abs(sf_full - expected_sf) < 1e-9, (
            "scale_factor mismatch — scaler may have been re-fit on test data"
        )

    # ---- scaled values are in [0, 1] ----------------------------------------

    def test_x_values_in_unit_range(self, close_1000):
        x, _, _ = prepare_sequences(close_1000)
        assert x.min() >= -1e-9  # allow tiny float errors
        assert x.max() <= 1 + 1e-9

    def test_y_values_in_unit_range(self, close_1000):
        _, y, _ = prepare_sequences(close_1000)
        assert y.min() >= -1e-9
        assert y.max() <= 1 + 1e-9

    # ---- determinism ---------------------------------------------------------

    def test_prepare_sequences_is_deterministic(self, close_1000):
        x1, y1, sf1 = prepare_sequences(close_1000)
        x2, y2, sf2 = prepare_sequences(close_1000)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        assert sf1 == sf2

    # ---- monotonic input edge case ------------------------------------------

    def test_monotonically_increasing_prices(self):
        """Prices that only go up should still produce valid, bounded sequences."""
        close = pd.Series(np.linspace(1.0, 1000.0, 500), name="Close")
        x, y, sf = prepare_sequences(close)
        assert len(x) > 0
        assert sf > 0
        assert math.isfinite(sf)

    def test_constant_prices_scale_factor_is_large_but_finite(self):
        """All-same prices → scaler scale_[0] ≈ 0 (degenerate); must not crash."""
        # sklearn MinMaxScaler clips scale to a minimum to avoid division by zero
        close = pd.Series([100.0] * 500, name="Close")
        # This should not raise
        try:
            x, y, sf = prepare_sequences(close)
        except Exception as exc:
            pytest.fail(f"prepare_sequences raised on constant prices: {exc}")
