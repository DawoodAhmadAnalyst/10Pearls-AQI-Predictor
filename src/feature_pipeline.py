import os
import numpy as np
import pandas as pd
import requests
import hopsworks
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAT, LON = 30.1575, 71.5249  # Multan
TIMEZONE = "Asia/Karachi"

WEATHER_HOURLY_PARAMS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,cloud_cover,precipitation"
AIR_QUALITY_HOURLY_PARAMS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust"

LAG_HOURS = [1, 3, 6, 12, 24]
TARGET_HORIZONS = [24, 48, 72]  # 24h/48h/72h ahead, matches Feature Group v2

# Live pipeline rolling window: must cover the longest target horizon (72h)
# plus the longest lag lookback (24h), with a safety buffer for missed runs
# or Open-Meteo data delays. 6 days (144h) gives a ~48h buffer above the
# 96h bare minimum (72 + 24).
LIVE_PAST_DAYS = 6

FEATURE_GROUP_NAME = "aqi_features_multan"
FEATURE_GROUP_VERSION = 2


# ---------------------------------------------------------------------------
# Data fetching — historical (archive API, used by backfill)
# ---------------------------------------------------------------------------

def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical weather data from Open-Meteo's archive API.
    Used for backfill only -- has a data-finalization lag of a few days,
    unsuitable for the live/hourly pipeline.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_HOURLY_PARAMS,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params).json()
    return pd.DataFrame(resp["hourly"])


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical air quality data from Open-Meteo. Backfill only."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_HOURLY_PARAMS,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params).json()
    return pd.DataFrame(resp["hourly"])


# ---------------------------------------------------------------------------
# Data fetching — recent/rolling (forecast API with past_days, used live)
# ---------------------------------------------------------------------------

def fetch_recent_weather(past_days: int = LIVE_PAST_DAYS) -> pd.DataFrame:
    """Fetch a recent rolling window of weather data via the forecast API's
    past_days parameter. Avoids the archive API's multi-day finalization lag,
    which makes it unsuitable for near-real-time hourly runs.

    forecast_days=1 (not 0): Open-Meteo treats "today" as forecast-day-0, so
    forecast_days=0 excludes today's already-elapsed hours entirely, not just
    future ones. forecast_days=1 includes today (both its real, already-
    elapsed hours and its not-yet-elapsed forecasted hours). The actual
    future-hours cutoff is enforced explicitly in run_live_pipeline via a
    timestamp filter, not left to this parameter alone.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": WEATHER_HOURLY_PARAMS,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params).json()
    return pd.DataFrame(resp["hourly"])


def fetch_recent_air_quality(past_days: int = LIVE_PAST_DAYS) -> pd.DataFrame:
    """Fetch a recent rolling window of air quality data. See fetch_recent_weather
    for why forecast_days=1 (not 0) is used here.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": AIR_QUALITY_HOURLY_PARAMS,
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params).json()
    return pd.DataFrame(resp["hourly"])


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_sources(df_weather: pd.DataFrame, df_air_quality: pd.DataFrame) -> pd.DataFrame:
    """Merge weather and air quality dataframes on the shared 'time' column.
    Drops 'ammonia' if present (validated: 100% null for Multan coordinates).
    """
    df_air_quality = df_air_quality.drop(columns=["ammonia"], errors="ignore")
    df = pd.merge(df_weather, df_air_quality, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def enforce_us_aqi_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Cast us_aqi to int64 to match the Feature Group schema (bigint).

    Must only be called AFTER NaN-dropping: the forecast API (live pipeline)
    can return NaN for us_aqi on the most recent hour(s) -- the air-quality
    forecast sometimes lags slightly behind the weather forecast -- and a
    column containing NaN cannot be cast to a non-nullable int type. Once
    NaN rows have been dropped, this is a safe, lossless cast (AQI is always
    a whole number; rounding first only guards against float representation
    noise like 87.99999999).
    """
    df = df.copy()
    df["us_aqi"] = df["us_aqi"].round().astype("int64")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + cyclical time features. See notebook 02 for EDA justification."""
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["day_of_year"] = df["time"].dt.dayofyear
    df["day_of_week"] = df["time"].dt.dayofweek

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_lag_features(df: pd.DataFrame, lag_hours=LAG_HOURS) -> pd.DataFrame:
    """Lag features for AQI, wind speed, and surface pressure.
    Motivated by the June 13 EDA finding: dust/AQI spikes lagged behind
    the wind event that caused them, so same-hour features alone miss this.
    """
    df = df.copy()
    for lag in lag_hours:
        df[f"aqi_lag_{lag}h"] = df["us_aqi"].shift(lag)
        df[f"wind_speed_lag_{lag}h"] = df["wind_speed_10m"].shift(lag)
        df[f"pressure_lag_{lag}h"] = df["surface_pressure"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean/std (smoothed trend + volatility) and change-rate (momentum)."""
    df = df.copy()
    df["aqi_roll_mean_6h"] = df["us_aqi"].rolling(window=6).mean()
    df["aqi_roll_mean_24h"] = df["us_aqi"].rolling(window=24).mean()
    df["aqi_roll_std_24h"] = df["us_aqi"].rolling(window=24).std()
    df["aqi_change_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_6h"] = df["us_aqi"].diff(6)
    return df


def add_targets(df: pd.DataFrame, horizons=TARGET_HORIZONS) -> pd.DataFrame:
    """Forward-shifted targets for each horizon (24h/48h/72h), aligned to the
    current row's features. NaN for the trailing rows of each horizon where
    the future value doesn't exist yet -- these are exactly the "live" rows
    that get self-healed by a later run once real data catches up to them.
    """
    df = df.copy()
    for h in horizons:
        df[f"target_aqi_{h}h"] = df["us_aqi"].shift(-h)
    return df


def engineer_features(df: pd.DataFrame, add_target_cols: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline, applied in order.
    Set add_target_cols=False only when you explicitly don't want target
    columns at all (rare -- the live pipeline still wants them, since the
    NaN targets are what get self-healed on later runs).
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    if add_target_cols:
        df = add_targets(df)
    return df


# ---------------------------------------------------------------------------
# Orchestration -- backfill (historical, one-off / manual)
# ---------------------------------------------------------------------------

def build_training_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """Full pipeline for a historical date range: fetch -> merge -> engineer -> clean.
    Used for the initial/manual backfill, not the hourly live pipeline.
    """
    df_weather = fetch_weather(start_date=start_date, end_date=end_date)
    df_air = fetch_air_quality(start_date=start_date, end_date=end_date)
    df = merge_sources(df_weather, df_air)
    df = engineer_features(df, add_target_cols=True)

    # Full backfill: drop rows missing ANY feature or ANY target -- there's
    # no "self-healing" concept here since this is a one-off historical load.
    df_clean = df.dropna().reset_index(drop=True)
    df_clean = enforce_us_aqi_dtype(df_clean)
    return df_clean


# ---------------------------------------------------------------------------
# Orchestration -- live pipeline (hourly, self-healing targets)
# ---------------------------------------------------------------------------

def run_live_pipeline(fg, past_days: int = LIVE_PAST_DAYS) -> pd.DataFrame:
    """Fetch a recent rolling window, engineer features + targets, and
    upsert into the Hopsworks Feature Group.

    Self-healing target mechanism: the Feature Group's primary key is
    'time'. A row inserted "now" will have NaN targets, since the future
    AQI doesn't exist yet. 24/48/72 hours later, that same 'time' value
    falls inside a NEW run's fetch window -- except now the actual future
    AQI is available, so the target can be computed. HUDI upserts on
    primary key, so the old NaN-target row is overwritten with the same
    features plus the now-computable target(s). No separate backfill/
    reconciliation job is needed.

    Rows are only dropped here if a FEATURE is missing (i.e. the leading
    rows of the window that don't have enough lag history yet) -- NaN
    targets are kept intentionally, since those are exactly the rows this
    mechanism is designed to heal on a future run.
    """
    df_weather = fetch_recent_weather(past_days=past_days)
    df_air = fetch_recent_air_quality(past_days=past_days)
    df = merge_sources(df_weather, df_air)

    # Explicit safeguard: drop any row later than right now, regardless of
    # how Open-Meteo buckets "today" vs "forecast" under forecast_days.
    # tz_localize(None) keeps this naive, consistent with df['time'] (which
    # is naive throughout this pipeline, matching all previously backfilled
    # data) -- comparing naive to tz-aware would otherwise raise a TypeError.
    now = pd.Timestamp.utcnow().tz_localize(None)
    df = df[df["time"] <= now].reset_index(drop=True)

    df = engineer_features(df, add_target_cols=True)

    target_cols = [f"target_aqi_{h}h" for h in TARGET_HORIZONS]
    feature_cols = [c for c in df.columns if c not in (["time"] + target_cols)]

    df_clean = df.dropna(subset=feature_cols).reset_index(drop=True)
    df_clean = enforce_us_aqi_dtype(df_clean)

    fg.insert(df_clean, write_options={"wait_for_job": False})
    return df_clean


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows-specific fix: no-op on GitHub Actions' Linux runners, required
    # locally since Hopsworks hardcodes /tmp internally even with stream=False.
    os.makedirs(r"D:\tmp", exist_ok=True)

    project = hopsworks.login(
        api_key_value=os.getenv("AQI_Predictor_KEY"),
        cert_folder="./hopsworks-certs",
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    df_result = run_live_pipeline(fg)
    print(f"Inserted {df_result.shape[0]} rows into {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
    print(f"Window: {df_result['time'].min()} to {df_result['time'].max()}")