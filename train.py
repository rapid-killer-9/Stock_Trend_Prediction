#!/usr/bin/env python3
"""
Train (or retrain) the LSTM stock-trend model and save it to disk.

The model is a 4-layer stacked LSTM with dropout, trained on the closing-price
series of any Yahoo Finance ticker.  Input shape matches the inference code in
app.py: (LOOKBACK_WINDOW, 1) = (100, 1).

Usage
-----
    python train.py --ticker AAPL
    python train.py --ticker SBIN.NS --years 15 --epochs 100 --output Keras_Model.h5

Arguments
---------
    --ticker      Yahoo Finance ticker symbol (required)
    --years       Years of history to download (default: 10)
    --epochs      Maximum training epochs; early-stopping may halt sooner (default: 50)
    --batch-size  Mini-batch size (default: 32)
    --output      Destination path for the saved model (default: Keras_Model.h5)
"""

import argparse
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.layers import Dense, Dropout, LSTM
from tf_keras.models import Sequential

# Mirror the constants used in app.py so inference is consistent.
LOOKBACK_WINDOW = 100
TRAIN_SPLIT = 0.70
DEFAULT_OUTPUT = "Keras_Model.h5"


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(input_shape: tuple) -> Sequential:
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


# ── Data helpers ──────────────────────────────────────────────────────────────
def build_sequences(
    series: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(window, len(series)):
        x.append(series[i - window : i])
        y.append(series[i, 0])
    return np.array(x), np.array(y)


def download_and_split(
    ticker: str, start: date, end: date
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    print(f"Downloading {ticker} from {start} to {end}…")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        print(f"ERROR: no data returned for '{ticker}'.", file=sys.stderr)
        sys.exit(1)

    close = df["Close"].values.reshape(-1, 1)
    split_idx = int(len(close) * TRAIN_SPLIT)
    train_raw = close[:split_idx]

    if len(train_raw) <= LOOKBACK_WINDOW:
        print(
            f"ERROR: training set ({len(train_raw)} rows) is too small for "
            f"the lookback window ({LOOKBACK_WINDOW}).",
            file=sys.stderr,
        )
        sys.exit(1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw)

    x_train, y_train = build_sequences(train_scaled, LOOKBACK_WINDOW)
    print(
        f"Rows total: {len(close)} | "
        f"Train: {len(train_raw)} | "
        f"Sequences: {len(x_train)}"
    )
    return x_train, y_train, scaler


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM stock-trend model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Yahoo Finance ticker symbol")
    parser.add_argument(
        "--years", type=int, default=10, help="Years of history to train on"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Max training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output .h5 model path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start = date.today() + relativedelta(years=-args.years)
    end = date.today()

    x_train, y_train, _ = download_and_split(args.ticker, start, end)

    model = build_model((LOOKBACK_WINDOW, 1))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(args.output, save_best_only=True, monitor="val_loss"),
    ]

    print(f"\nTraining for up to {args.epochs} epochs…")
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"\nModel saved → {args.output}")


if __name__ == "__main__":
    main()
