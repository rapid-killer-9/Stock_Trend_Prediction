# Stock Trend Prediction

A Streamlit web app that visualises historical stock prices and overlays LSTM-based trend predictions on any Yahoo Finance–listed ticker.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI / server | [Streamlit](https://streamlit.io/) 1.12 |
| Data source | [yfinance](https://github.com/ranaroussi/yfinance) 0.2 |
| ML model | TensorFlow / Keras 2.9 (pre-trained LSTM, `Keras_Model.h5`) |
| Preprocessing | scikit-learn `MinMaxScaler` |
| Charting | Matplotlib 3.5 |
| Language | Python 3.9+ |
| Dev container | GitHub Codespaces / VS Code Dev Containers |

---

## Key Features

- **LSTM price prediction** — loads a pre-trained Keras model and plots predicted vs. actual closing prices for the test split of the selected period.
- **Retrainable model** — `train.py` CLI lets you retrain the LSTM on any ticker and period; the saved weights are a drop-in replacement for `Keras_Model.h5`.
- **CSV export** — download the full OHLCV dataset or the aligned (Date, Actual, Predicted) prediction table directly from the UI.
- **Interactive time range** — choose 1, 2, 3, 5, 10, or 20 years of history from a dropdown; data is fetched live from Yahoo Finance.
- **Moving-average overlays** — 100-day MA and 200-day MA charts give a technical-analysis baseline alongside the raw closing price.
- **Input validation & error recovery** — empty tickers are blocked before any network call; unrecognised tickers surface an actionable error with a link to Yahoo Finance.
- **Request caching** — `@st.cache_data` on the data-fetch call and `@st.cache_resource` on model loading prevent redundant work across Streamlit reruns.

---

## Architecture

```
Stock_Trend_Prediction/
├── app.py               # Streamlit app — UI, data fetch, inference, CSV export
├── train.py             # CLI training pipeline — build and save a new LSTM model
├── Keras_Model.h5       # Pre-trained LSTM weights loaded by app.py at runtime
├── requirements.txt     # Pinned Python dependencies
├── .env.example         # Environment variable template (no secrets required currently)
├── .devcontainer/
│   └── devcontainer.json  # Codespaces / VS Code Dev Container config
├── Images/              # Screenshots used in documentation
└── LICENSE              # MIT
```

### Request flow

```
User input (ticker, date range)
    │
    ▼
start_date_for_range()          ← maps UI label → Python date
    │
    ▼
fetch_stock_data()              ← yfinance.download(), cached with @st.cache_data
    │
    ├─► describe() + MA charts  ← rendered immediately with Matplotlib / st.pyplot
    │
    ▼
prepare_sequences()             ← 70/30 split, MinMaxScaler, 100-day sliding window
    │
    ▼
load_keras_model()              ← Keras_Model.h5, cached with @st.cache_resource
    │
    ▼
model.predict()
    │
    ▼
Inverse-scale predictions → st.pyplot (Predicted vs Actual chart)
```

---

## Design Decisions

- **Pre-trained model shipped in the repo** — the LSTM (`Keras_Model.h5`) was trained offline; `app.py` performs inference only. This keeps the Streamlit app fast and stateless, with no training dependency at runtime.
- **70 / 30 train-test split at prediction time** — the first 70 % of the requested history seeds the scaler; the final 30 % is used as the test window. The 100-day tail of the training set is prepended to the test set to provide the model's lookback context.
- **MinMaxScaler fit on training slice only** — the scaler is fitted exclusively on training data before being applied to the test window, which prevents data leakage from future prices into the normalisation step.
- **`@st.cache_resource` for the Keras model** — Keras models are not picklable in all versions, so `cache_resource` (which stores the live Python object rather than serialising it) is used instead of `cache_data`.

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- `pip`

### Install

```bash
git clone https://github.com/<your-username>/Stock_Trend_Prediction.git
cd Stock_Trend_Prediction
pip install -r requirements.txt
```

### Environment variables

No API keys are required for basic usage. Copy the example file for future configuration:

```bash
cp .env.example .env
```

### Run

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

### GitHub Codespaces / Dev Container

Open the repo in a Codespace or VS Code with the Dev Containers extension — the container installs dependencies and starts the server automatically on port 8501.

---

## Testing & API Docs

> **No automated tests exist yet.** Manual testing steps:
>
> 1. Enter a valid ticker (e.g. `AAPL`, `SBIN.NS`) and verify charts render.
> 2. Enter an invalid ticker and verify the error message appears without a traceback.
> 3. Select "1-Year" for a recently listed stock and verify the short-history warning fires.
>
> There is no REST API or Swagger/Postman collection — the app is a self-contained Streamlit UI.

---

## Retraining the model

`train.py` trains a new LSTM on any Yahoo Finance ticker and saves the weights to `Keras_Model.h5` (or a custom path), ready to be loaded by `app.py`.

```bash
# Retrain on Apple, 10 years of history, default 50 epochs
python train.py --ticker AAPL

# More control
python train.py --ticker SBIN.NS --years 15 --epochs 100 --output my_model.h5
```

| Flag | Default | Description |
|---|---|---|
| `--ticker` | *(required)* | Yahoo Finance symbol |
| `--years` | `10` | Years of history to download |
| `--epochs` | `50` | Max training epochs (EarlyStopping may stop sooner) |
| `--batch-size` | `32` | Mini-batch size |
| `--output` | `Keras_Model.h5` | Destination path for saved weights |

**Model architecture** — 4-layer stacked LSTM (50 units) with 0.2 dropout after each LSTM layer, Dense(25), Dense(1). Input shape `(100, 1)` matches `LOOKBACK_WINDOW` in `app.py`.

---

## Exporting data

Two download buttons appear in the app:

- **Download raw data (CSV)** — full OHLCV DataFrame for the selected ticker and period.
- **Download predictions (CSV)** — three-column file (`Date`, `Actual Price`, `Predicted Price`) for the test window.

---

## Future Improvements

- **Automated tests with `pytest`** — unit-test `prepare_sequences()` and `start_date_for_range()` in isolation; add a smoke test that mocks `yfinance.download` and asserts the prediction chart renders without error.
- **Move `Keras_Model.h5` to Git LFS or a model registry** — binary ML weights bloat the repo history; a dedicated artifact store (e.g. DVC, Hugging Face Hub) is more appropriate for large model files.

---

## License

[MIT](LICENSE)
