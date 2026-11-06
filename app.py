import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH = "Keras_Model.h5"
TRAIN_SPLIT = 0.70
LOOKBACK_WINDOW = 100

DATE_RANGE_OPTIONS = ["10-Year", "1-Year", "2-Year", "3-Year", "5-Year", "20-Year"]

_DELTA_MAP = {
    "1-Year": relativedelta(years=-1),
    "2-Year": relativedelta(years=-2),
    "3-Year": relativedelta(years=-3),
    "5-Year": relativedelta(years=-5),
    "10-Year": relativedelta(years=-10),
    "20-Year": relativedelta(years=-20),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def start_date_for_range(selected: str) -> date:
    delta = _DELTA_MAP.get(selected)
    if delta is None:
        return date.today()
    return date.today() + delta


@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    return df


@st.cache_resource(show_spinner=False)
def load_keras_model():
    return load_model(MODEL_PATH)


def prepare_sequences(
    close: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split close prices into scaled test sequences and return the scale factor."""
    split_idx = int(len(close) * TRAIN_SPLIT)
    training = pd.DataFrame(close[:split_idx])
    testing = pd.DataFrame(close[split_idx:])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(training)

    seed_window = training.tail(LOOKBACK_WINDOW)
    combined = pd.concat([seed_window, testing], ignore_index=True)
    scaled = scaler.fit_transform(combined)

    x_test, y_test = [], []
    for i in range(LOOKBACK_WINDOW, scaled.shape[0]):
        x_test.append(scaled[i - LOOKBACK_WINDOW : i])
        y_test.append(scaled[i, 0])

    scale_factor = 1.0 / scaler.scale_[0]
    return np.array(x_test), np.array(y_test), scale_factor


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Stock Trend Prediction")

ticker = st.text_input("Enter Stock Ticker", "SBIN.NS").strip().upper()
selected_range = st.selectbox("Select Time Period", options=DATE_RANGE_OPTIONS)

start = start_date_for_range(selected_range)
end = date.today()

if not ticker:
    st.warning("Please enter a stock ticker to continue.")
    st.stop()

with st.spinner(f"Fetching data for {ticker}…"):
    data = fetch_stock_data(ticker, start, end)

if data.empty:
    st.error(
        f"No data found for ticker **{ticker}**. "
        "Check the symbol on [Yahoo Finance](https://finance.yahoo.com/)."
    )
    st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────
st.write(f"Data from **{start}** to **{end}**")
st.write(data.describe())
st.download_button(
    label="Download raw data (CSV)",
    data=data.to_csv().encode("utf-8"),
    file_name=f"{ticker}_{start}_{end}.csv",
    mime="text/csv",
)

# ── Moving-average charts ─────────────────────────────────────────────────────
close = data["Close"].squeeze()

st.subheader("Closing Price vs Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(close, label="Close")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Closing Price with 100-day MA")
ma100 = close.rolling(LOOKBACK_WINDOW).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ma100, "r", label="100-day MA")
ax.plot(close, "b", label="Close")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Closing Price with 100-day & 200-day MA")
ma200 = close.rolling(200).mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ma100, "r", label="100-day MA")
ax.plot(ma200, "g", label="200-day MA")
ax.plot(close, "b", label="Close")
ax.legend()
st.pyplot(fig)
plt.close(fig)

# ── Prediction ────────────────────────────────────────────────────────────────
with st.spinner("Running prediction model…"):
    model = load_keras_model()
    x_test, y_test, scale_factor = prepare_sequences(close)

    if len(x_test) == 0:
        st.warning(
            "Not enough data to run predictions for the selected time period. "
            "Try a longer range."
        )
        st.stop()

    y_predicted = model.predict(x_test, verbose=0)

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Align test-window dates with predicted values for the export
split_idx = int(len(close) * TRAIN_SPLIT)
test_dates = close.index[split_idx : split_idx + len(y_test)]

st.subheader("Predicted vs Actual Closing Price")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(y_test, "b", label="Actual Price")
ax.plot(y_predicted, "r", label="Predicted Price")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
plt.close(fig)

results_df = pd.DataFrame(
    {
        "Date": test_dates,
        "Actual Price": y_test,
        "Predicted Price": y_predicted.flatten(),
    }
)
st.download_button(
    label="Download predictions (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ticker}_predictions_{start}_{end}.csv",
    mime="text/csv",
)
