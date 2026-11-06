# Stock Trend Prediction

A Streamlit app that fetches live OHLCV data from Yahoo Finance and overlays LSTM-based closing-price predictions on any listed ticker.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI / server | Streamlit 1.22 |
| Data source | yfinance ≥ 0.2.40 (Yahoo Finance) |
| ML framework | TensorFlow / Keras 2.9 |
| Preprocessing | scikit-learn `MinMaxScaler` |
| Charting | Matplotlib 3.5 |
| Language | Python 3.10 |
| Container | Docker — `python:3.10-slim`, venv at `/opt/venv` |
| Dev container | GitHub Codespaces / VS Code Dev Containers |

---

## Key Features

- **Live data fetch** — downloads OHLCV history for any Yahoo Finance ticker across six selectable time ranges (1–20 years); results are cached per session with `@st.cache_data`.
- **LSTM inference** — loads a pre-trained 4-layer stacked LSTM (`Keras_Model.h5`) and predicts the closing-price series over the test window; the model is cached in-process with `@st.cache_resource`.
- **Technical overlays** — plots raw closing price, 100-day MA, and 200-day MA as separate charts for quick visual analysis.
- **Retrainable model** — `train.py` CLI retrains the LSTM on any ticker and period with early stopping and best-weights checkpointing; the output `.h5` is a drop-in replacement for the bundled weights.
- **CSV export** — two download buttons: full OHLCV dataset and an aligned `(Date, Actual Price, Predicted Price)` table for the test window.
- **Input validation** — empty tickers are blocked before any network call; tickers that return no data surface an actionable error with a link to Yahoo Finance.

---

## Architecture

```
Stock_Trend_Prediction/
├── app.py                  # Streamlit app: UI, data fetch, inference, charts, CSV export
├── train.py                # CLI: download data, train LSTM, save weights
├── Keras_Model.h5          # Pre-trained weights (loaded at inference time)
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Docker image: python:3.10-slim + venv at /opt/venv
├── .dockerignore           # Excludes venv/, Images/, .git/ from build context
├── .env.example            # Env-var template (no secrets currently required)
├── .devcontainer/
│   └── devcontainer.json   # Codespaces / VS Code Dev Container config
├── Images/                 # Screenshots for documentation
└── LICENSE                 # MIT
```

### Request flow

```
User selects ticker + date range
        │
        ▼
start_date_for_range()        maps UI label → Python date via relativedelta
        │
        ▼
fetch_stock_data()            yf.download()  — @st.cache_data (keyed on ticker + dates)
        │
        ├──► data.describe()  summary stats table
        ├──► Closing price chart           ┐
        ├──► 100-day MA chart              ├─ matplotlib → st.pyplot()
        ├──► 100-day + 200-day MA chart    ┘
        │
        ▼
prepare_sequences()
  ├── 70 / 30 split on close prices
  ├── MinMaxScaler.fit_transform() on training slice only
  ├── prepend 100-day tail of training set to test set
  └── build sliding-window sequences of shape (100, 1)
        │
        ▼
load_keras_model()            Keras_Model.h5 — @st.cache_resource
        │
        ▼
model.predict(x_test)
        │
        ▼
inverse-scale → Predicted vs Actual chart + CSV download
```

---

## Design Decisions

- **Inference-only app** — `Keras_Model.h5` is trained offline via `train.py` and checked into the repo. `app.py` performs inference only, keeping the Streamlit deployment fast and stateless with no training dependency at runtime.
- **Scaler fit on training slice only** — `MinMaxScaler` is fitted exclusively on the first 70 % of the requested history before being applied to the test window. Fitting on all data would leak future price information into the normalisation step.
- **100-day seed window injected into test set** — the tail of the training slice is prepended to the test data so the model always has a full 100-day lookback context for its first prediction, matching the `LOOKBACK_WINDOW` used during training.
- **`@st.cache_resource` for the Keras model** — Keras objects are not reliably serialisable across all versions, so `cache_resource` (stores the live Python object) is used instead of `cache_data` (which requires serialisation).

---

## CLI Reference (`train.py`)

There is no REST API — the app is a self-contained Streamlit UI. The only scriptable interface is the training CLI.

```bash
python train.py --ticker AAPL
python train.py --ticker SBIN.NS --years 15 --epochs 100 --output my_model.h5
```

| Flag | Default | Description |
|---|---|---|
| `--ticker` | *(required)* | Yahoo Finance symbol |
| `--years` | `10` | Years of history to download |
| `--epochs` | `50` | Max training epochs (EarlyStopping patience=5 may stop sooner) |
| `--batch-size` | `32` | Mini-batch size |
| `--output` | `Keras_Model.h5` | Destination path for saved weights |

---

## Data Shape

No database is used. All data is fetched at runtime from Yahoo Finance.

| Stage | Shape | Notes |
|---|---|---|
| Raw download | `(n_days, 6)` | OHLCV + Adj Close from yfinance |
| Close series | `(n_days,)` | `data["Close"].squeeze()` |
| Training slice | `(n_days × 0.70, 1)` | MinMaxScaler fitted here |
| x_train / x_test sequences | `(n_sequences, 100, 1)` | 100-day sliding window |
| Model output | `(n_sequences, 1)` | Scaled predictions, then inverse-scaled |

**Model architecture** (`build_model` in `train.py`):

```
LSTM(50, return_sequences=True) → Dropout(0.2)
LSTM(50, return_sequences=True) → Dropout(0.2)
LSTM(50)                        → Dropout(0.2)
Dense(25) → Dense(1)
Optimizer: Adam   Loss: MSE
```

---

## Getting Started

### Option A — Docker (recommended)

Pins Python 3.10 and isolates all packages inside `/opt/venv`, so your host Python version and future package updates cannot break the app.

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/)

**Step 1 — Clone the repository**
```bash
git clone https://github.com/<your-username>/Stock_Trend_Prediction.git
cd Stock_Trend_Prediction
```

**Step 2 — Build the Docker image**
```bash
docker build -t stock-trend-prediction .
#                                       ^ build context: current directory (required)
```

**Step 3 — Run the container**
```bash
docker run -p 8501:8501 stock-trend-prediction
```

**Step 4 — Open the app**

Visit [http://localhost:8501](http://localhost:8501).

To run in the background:
```bash
docker run -d -p 8501:8501 --name stp stock-trend-prediction
docker logs -f stp    # stream logs
docker stop stp       # stop when done
```

#### Image layout

| Path | Contents |
|---|---|
| `/usr/local/` | System Python 3.10 (untouched) |
| `/opt/venv/` | App venv — all packages from `requirements.txt` |
| `/app/` | `app.py`, `train.py`, `Keras_Model.h5` |

---

### Option B — Local venv

**Prerequisites:** Python 3.10 exactly (`python3.10 --version`)

> **Note:** Python 3.11+ is not supported — `numpy==1.23.2` and `tensorflow==2.9.1` have no pre-built wheels for newer interpreters and will fail to build from source.

```bash
git clone https://github.com/<your-username>/Stock_Trend_Prediction.git
cd Stock_Trend_Prediction

python3.10 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

### Environment variables

No API keys are required. Copy the example file for future configuration:

```bash
cp .env.example .env
```

### GitHub Codespaces / Dev Container

Open the repo in a Codespace or VS Code with the Dev Containers extension — the container installs dependencies and starts the server automatically on port 8501.

> **Note:** The devcontainer image uses Python 3.11. If you hit dependency errors, switch to Option A (Docker) or Option B (local Python 3.10 venv).

---

## Retraining the model

```bash
# Inside the venv (Option B) or a docker exec shell:
python train.py --ticker AAPL
```

The script downloads history, trains with early stopping, and saves the best checkpoint to `Keras_Model.h5` (or the path given with `--output`). Place the file in the repo root; `app.py` loads it on the next deploy.

---

## Testing

No automated tests exist. Manual smoke-test steps:

1. Enter a valid ticker (`AAPL`, `SBIN.NS`) and verify all three MA charts and the prediction chart render.
2. Enter an invalid ticker (`INVALID`) and verify the error message appears with a Yahoo Finance link.
3. Select a short range (1-Year) for a recently listed stock and verify the "not enough data" warning fires.

---

## Future Improvements

- **Add a pytest suite** — unit-test `prepare_sequences()` (fewer than 100 rows, single-day input) and `start_date_for_range()` with a mocked `date.today()`; smoke-test the prediction path with a mocked `yf.download`.
- **Move `Keras_Model.h5` out of git** — binary weights bloat repo history; store in Git LFS, DVC, or Hugging Face Hub and fetch at container build time.
- **Fix the shifting train/test boundary** — the 70/30 split is recalculated from the selected date range at inference time, so the test window shown in the UI may not correspond to the data the shipped model was actually trained on; make the split date a fixed parameter.

---

## License

[MIT](LICENSE)
