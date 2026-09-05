# Pearls AQI Predictor

A serverless, end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Multan, Pakistan, 24, 48, and 72 hours ahead. Built as a Data Science internship project at 10Pearls Pakistan.

Live weather and air-quality data is pulled hourly from Open-Meteo, engineered into 47 features, and stored in a Hopsworks Feature Store. A tuned Ridge Regression model — one per forecast horizon — is retrained daily and served through a Streamlit dashboard, with the whole pipeline automated by GitHub Actions and no persistently-running infrastructure owned by the project itself.

For the full write-up — EDA findings, model comparison, the data-quality issues hit along the way, and what was learned — see `Reports/`.

## Architecture

```
Open-Meteo (weather + air quality)
        │  hourly
        ▼
src/feature_pipeline.py  ──────►  Hopsworks Feature Store
  (fetch, engineer,                (aqi_features_multan, v2)
   upsert)                                  │  daily
                                             ▼
                                   src/train_pipeline.py
                                    (train, evaluate,
                                     register 3 Ridge models)
                                             │
                                             ▼
                                   Hopsworks Model Registry
                                   (aqi_ridge_24h/48h/72h)
                                             │
                                             ▼
                                   app.py (Streamlit dashboard)
                                   (forecast, SHAP, hazard alert)
```

Both pipeline stages run on GitHub Actions' schedule — nothing here needs a server you keep running yourself.

## Repository structure

```
.
├── app.py                        # Streamlit dashboard (entry point)
├── requirements.txt               # Python dependencies
├── packages.txt                   # apt packages needed on Streamlit Community Cloud
├── src/
│   ├── feature_pipeline.py        # Hourly: fetch → engineer → upsert to Feature Store
│   └── train_pipeline.py          # Daily: train, evaluate, register 3 Ridge models
├── utils/
│   ├── hopsworks_utils.py         # Hopsworks connection + Feature Store reads
│   ├── prediction.py              # Model loading (cached) + inference
│   └── aqi_utils.py               # AQI category/color/emoji lookup
├── Notebooks/
│   ├── 1-EDA.ipynb                # Seasonality, correlations, outlier investigation
│   ├── 2-Feature Engineering.ipynb # Feature design + Feature Store write
│   └── 3-Training.ipynb           # Model comparison, tuning, final training
├── Data/
│   └── sample_data.csv            # Small local sample for offline notebook work
├── Reports/                        # Final internship report
├── HopsWork/                       # Hopsworks-side notes/exports (screenshots, etc.)
└── .github/workflows/
    ├── feature_pipeline.yml        # Runs src/feature_pipeline.py hourly
    └── train_pipeline.yml          # Runs src/train_pipeline.py daily at 02:00 UTC
```

Not committed to git (see `.gitignore`), but present locally after running the pipelines: `hopsworks-certs/` (downloaded automatically on login), `*.pkl` model bundles, `models/`, `.env`. These are regenerated automatically — the Feature Store and Model Registry are the actual source of truth, not anything sitting in the repo.

## Data

- **Weather** and **air quality** both come from [Open-Meteo](https://open-meteo.com/) — coordinate-based (30.1575° N, 71.5249° E), no API key required.
  - Historical/backfill: the archive API (`archive-api.open-meteo.com`)
  - Live hourly runs: the forecast API's `past_days` parameter (`api.open-meteo.com`), since the archive API has a multi-day finalization lag unsuitable for near-real-time use
- **Target**: `us_aqi` (US EPA methodology), forecast at 24h/48h/72h via `target_aqi_24h/48h/72h`
- Ammonia is dropped (100% null for these coordinates); pollen was never included (Europe-only coverage)

## Feature Store design

Feature Group: `aqi_features_multan`, version 2 (primary key and event-time column: `time`, HUDI format).

The three targets are just forward-shifted AQI values, so the most recent rows in any hourly write genuinely can't have a real target yet. Those rows go in with `NaN` targets; as later runs' fetch windows reach those same timestamps with real, observed AQI, HUDI's primary-key upsert overwrites the null with the real value — no separate backfill or reconciliation job required. `train_pipeline.py` filters out any row with a future timestamp before computing the train/test split, as a safety net against this self-healing mechanism ever leaking un-healed data into training.

A version 1 of this Feature Group also exists in Hopsworks — an early, single-horizon prototype schema from before the self-healing design was finalized. It's kept only as a historical artifact; nothing in this codebase reads from it.

## Models

Three independently-trained Ridge Regression models (`alpha=10.0`, `StandardScaler`-normalized features), one per horizon, registered in the Hopsworks Model Registry as `aqi_ridge_24h`, `aqi_ridge_48h`, `aqi_ridge_72h`. Each is bundled as a single joblib artifact containing the model, its scaler, and its exact training-time feature column list, so the dashboard can never accidentally apply a model to mismatched features.

Ridge was chosen over Random Forest and XGBoost after both tree models showed strong cross-validation scores that didn't hold up against a genuinely held-out, chronological test set — see `Reports/` for the full comparison and final metrics.

## Setup

**Requirements:** Python 3.11, a Hopsworks account (free tier) with a project and API key.

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the project root (never commit this):
   ```
   AQI_Predictor_KEY=your_hopsworks_api_key
   ```
3. Run the feature pipeline once manually to populate the Feature Store, then the training pipeline to register models:
   ```
   python src/feature_pipeline.py
   python src/train_pipeline.py
   ```
4. Run the dashboard locally:
   ```
   streamlit run app.py
   ```

**On Windows**, `hopsworks.login()` needs an explicit `cert_folder` (already set to `./hopsworks-certs` in every script here), and Hopsworks' client hardcodes a reference to `/tmp` internally even for non-streaming writes — `feature_pipeline.py` and `train_pipeline.py` both create a `D:\tmp` directory for this reason. This is a no-op on GitHub Actions' Linux runners and on Streamlit Community Cloud, both of which use the OS's own temp directory instead (`utils/hopsworks_utils.py` handles this with `tempfile.gettempdir()`).

## Automation

- **`feature_pipeline.yml`** — runs hourly (`0 * * * *`), fetches the latest ~6 days of data (covering the longest lag and target horizon with a buffer for missed runs), engineers features, and upserts into the Feature Store.
- **`train_pipeline.yml`** — runs daily at 02:00 UTC, retrains and re-registers all three models against the latest Feature Store snapshot.

Both are also manually triggerable from the Actions tab (`workflow_dispatch`). The Hopsworks API key is stored as the GitHub Secret `AQI_PREDICTOR_KEY`.

## Deployment

The dashboard is deployed on Streamlit Community Cloud. `packages.txt` (`librdkafka-dev`) installs the system-level dependency Hopsworks' Kafka client needs on Streamlit's Linux containers — without it, `hopsworks[python]` fails to import there even though it works fine locally. On Streamlit Community Cloud, the Hopsworks API key is set via the app's **Secrets** (not `.env`); `utils/hopsworks_utils.py` checks both `os.getenv` and `st.secrets` so the same code works in both environments.

## Known limitations

- The February 20, 2024 low-AQI anomaly identified during EDA remains unexplained by the current feature set (see `Reports/`).
- An LSTM was considered but not built — no evidence it would outperform Ridge given how strongly seasonal (rather than sequential) the signal turned out to be.
- Hopsworks' free-tier job-status reporting isn't always reliable (a job can report `FAILED` due to an unrelated shutdown timeout after the actual write already succeeded) — worth independently verifying Feature Store state after any manual write, rather than trusting the job status alone.