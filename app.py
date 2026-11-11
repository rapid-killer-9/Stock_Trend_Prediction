import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import date
from tf_keras.models import load_model
import streamlit as st

from core import (
    DATE_RANGE_OPTIONS,
    LOOKBACK_WINDOW,
    MODEL_PATH,
    TRAIN_SPLIT,
    prepare_sequences,
    start_date_for_range,
)


@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)


@st.cache_resource(show_spinner=False)
def load_keras_model():
    return load_model(MODEL_PATH)


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
