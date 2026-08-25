import os
import shutil
import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_GROUP_NAME = "aqi_features_multan"
FEATURE_GROUP_VERSION = 2

TARGET_COLS = ["target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]
HORIZONS = ["24h", "48h", "72h"]

RIDGE_ALPHA = 10.0
TEST_MONTHS = 13  # holds out the last N months as test, to cover a full seasonal cycle

MODELS_DIR = "models"
REGISTRY_DIR_TEMPLATE = "models_registry_{horizon}"


# ---------------------------------------------------------------------------
# Hopsworks connection
# ---------------------------------------------------------------------------

def connect_to_hopsworks():
    """Login to Hopsworks and return (project, feature_store, model_registry).

    Includes the Windows-specific fixes required for this environment:
    - cert_folder override (default /tmp path doesn't exist on Windows)
    - D:\\tmp directory, since Hopsworks hardcodes /tmp internally for
      Kafka/Hudi even when stream=False
    """
    os.makedirs(r"D:\tmp", exist_ok=True)

    project = hopsworks.login(
        api_key_value=os.getenv("AQI_Predictor_KEY"),
        cert_folder="./hopsworks-certs",
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    return project, fs, mr


def load_feature_data(fs):
    """Read the v2 Feature Group and return a sorted dataframe.

    Sorting is required: Hopsworks reads are not guaranteed to preserve
    insertion order, and the chronological train/test split downstream
    depends on correct time ordering.
    """
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("time").reset_index(drop=True)

    # Safety filter: drop any row timestamped later than right now. Protects
    # against forecast-contaminated rows (e.g. from a live-pipeline bug that
    # briefly wrote forecasted, not observed, future data) inflating
    # df['time'].max() and skewing the split date, or sneaking synthetic
    # rows into the test set. Cheap insurance, worth keeping permanently.
    #
    # Data read back from Hopsworks comes back tz-aware (UTC) -- unlike the
    # raw Open-Meteo fetch in feature_pipeline.py, which is naive. "now" must
    # match that here, or pandas raises a tz-naive vs tz-aware TypeError.
    now = pd.Timestamp.now(tz="UTC")
    before = len(df)
    df = df[df["time"] <= now].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} row(s) with a future timestamp (> {now}) before training.")

    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_feature_cols(df):
    """All columns except the time index and the three target columns."""
    return [c for c in df.columns if c not in (["time"] + TARGET_COLS)]


def compute_split_date(df, test_months=TEST_MONTHS):
    """Fixed calendar-date split, shared across all three horizons.

    Using a date (not a row count) ensures all three models are evaluated
    on the identical test period, even though each horizon drops a
    different number of trailing NaN rows.
    """
    max_date = df["time"].max()
    return max_date - pd.DateOffset(months=test_months)


def train_ridge_for_horizon(df, target_col, feature_cols, split_date, alpha=RIDGE_ALPHA):
    """Train and evaluate a single Ridge model for one forecast horizon.

    NaN dropping happens here, independently per horizon:
    - trailing NaNs come from the target shift (different per horizon)
    - leading NaNs come from lag/rolling features needing prior history
      (same ~24 rows regardless of horizon)
    Both must be dropped before fitting, or Ridge will raise on NaN input.
    """
    data = df.dropna(subset=feature_cols + [target_col]).copy()

    train = data[data["time"] < split_date]
    test = data[data["time"] >= split_date]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return {
        "model": model,
        "scaler": scaler,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_train": len(train),
        "n_test": len(test),
    }


def train_all_horizons(df, feature_cols, split_date):
    """Train all three Ridge models and return a dict keyed by horizon."""
    results = {}
    for horizon, target_col in zip(HORIZONS, TARGET_COLS):
        results[horizon] = train_ridge_for_horizon(df, target_col, feature_cols, split_date)
        r = results[horizon]
        print(
            f"{horizon} -> RMSE: {r['rmse']:.2f}, MAE: {r['mae']:.2f}, "
            f"R²: {r['r2']:.3f}, n_train={r['n_train']}, n_test={r['n_test']}"
        )
    return results


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def save_and_register_models(mr, results, feature_cols):
    """Bundle each model with its scaler + feature_cols, then push to the
    Hopsworks Model Registry.

    Bundling model + scaler + feature_cols into one file avoids a whole
    class of bugs where the wrong scaler gets paired with the wrong model,
    or inference features get passed in the wrong column order.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    for horizon in HORIZONS:
        r = results[horizon]

        bundle = {
            "model": r["model"],
            "scaler": r["scaler"],
            "feature_cols": feature_cols,
        }
        bundle_path = os.path.join(MODELS_DIR, f"aqi_ridge_{horizon}.pkl")
        joblib.dump(bundle, bundle_path)

        model_dir = REGISTRY_DIR_TEMPLATE.format(horizon=horizon)
        os.makedirs(model_dir, exist_ok=True)
        shutil.copy(bundle_path, os.path.join(model_dir, f"aqi_ridge_{horizon}.pkl"))

        aqi_model = mr.sklearn.create_model(
            name=f"aqi_ridge_{horizon}",
            metrics={"rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"]},
            description=(
                f"Ridge regression for {horizon}-ahead AQI forecast (Multan), "
                f"scaled features, alpha={RIDGE_ALPHA}, bundled with "
                f"StandardScaler + feature_cols."
            ),
        )
        aqi_model.save(model_dir)
        print(
            f"Registered aqi_ridge_{horizon} — "
            f"RMSE={r['rmse']:.2f}, MAE={r['mae']:.2f}, R²={r['r2']:.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    project, fs, mr = connect_to_hopsworks()

    df = load_feature_data(fs)
    print(f"Loaded feature data: {df.shape}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")

    feature_cols = get_feature_cols(df)
    split_date = compute_split_date(df)
    print(f"Split date (start of test set): {split_date}")

    results = train_all_horizons(df, feature_cols, split_date)
    save_and_register_models(mr, results, feature_cols)

    print("Training pipeline complete — all three horizons registered.")


if __name__ == "__main__":
    main()