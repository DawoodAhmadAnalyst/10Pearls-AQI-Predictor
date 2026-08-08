"""
feature_pipeline.py

Core feature engineering logic for the Pearls AQI Predictor (Multan).
Used by BOTH the live (twice-hourly) pipeline and the historical backfill script,
so any fix/change here automatically applies to both.

Data source: Open-Meteo (Weather API + Air Quality API), coordinate-based.
"""

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LAT, LON = 30.1575, 71.5249  # Multan
TIMEZONE = "Asia/Karachi"

WEATHER_HOURLY_PARAMS = "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,cloud_cover,precipitation"
AIR_QUALITY_HOURLY_PARAMS = "us_aqi,pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust"

LAG_HOURS = [1, 3, 6, 12, 24]
TARGET_HORIZON_HOURS = 72  # 3 days ahead


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_weather(start_date: str = None, end_date: str = None, current: bool = False) -> pd.DataFrame:
    """
    Fetch weather data from Open-Meteo.

    If current=True, ignores start_date/end_date and fetches the latest
    forecast-endpoint reading (used by the live pipeline).
    Otherwise, fetches historical data for [start_date, end_date] (used by backfill).

    Dates should be strings in 'YYYY-MM-DD' format.
    """
    if current:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "current": WEATHER_HOURLY_PARAMS,
            "timezone": TIMEZONE,
        }
        resp = requests.get(url, params=params).json()
        # Wrap the single 'current' reading into a one-row dataframe for consistency
        row = resp["current"]
        return pd.DataFrame([row])
    else:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required when current=False")
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


def fetch_air_quality(start_date: str = None, end_date: str = None, current: bool = False) -> pd.DataFrame:
    """
    Fetch air quality data from Open-Meteo.
    Same current/historical pattern as fetch_weather().
    """
    if current:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        params = {
            "latitude": LAT,
            "longitude": LON,
            "current": AIR_QUALITY_HOURLY_PARAMS,
            "timezone": TIMEZONE,
        }
        resp = requests.get(url, params=params).json()
        row = resp["current"]
        return pd.DataFrame([row])
    else:
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required when current=False")
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


def merge_sources(df_weather: pd.DataFrame, df_air_quality: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather and air quality dataframes on the shared 'time' column.
    Drops 'ammonia' if present (validated: 100% null for Multan coordinates).
    """
    df_air_quality = df_air_quality.drop(columns=["ammonia"], errors="ignore")
    df = pd.merge(df_weather, df_air_quality, on="time", how="inner")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
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
    """
    Lag features for AQI, wind speed, and surface pressure.
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


def add_target(df: pd.DataFrame, horizon_hours: int = TARGET_HORIZON_HOURS) -> pd.DataFrame:
    """
    Forward-shifted target: AQI `horizon_hours` in the future, aligned to
    the current row's features. NaN for the last `horizon_hours` rows
    (no future data available yet) -- these are the rows used for live
    prediction, not training.
    """
    df = df.copy()
    df[f"target_aqi_{horizon_hours}h"] = df["us_aqi"].shift(-horizon_hours)
    return df


def engineer_features(df: pd.DataFrame, add_target_col: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline, applied in order.
    Set add_target_col=False when engineering features for a LIVE prediction row
    (no future data exists yet to build a target from).
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    if add_target_col:
        df = add_target(df)
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_training_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Full pipeline for a historical date range: fetch -> merge -> engineer -> clean.
    Used by backfill.py.
    """
    df_weather = fetch_weather(start_date=start_date, end_date=end_date, current=False)
    df_air = fetch_air_quality(start_date=start_date, end_date=end_date, current=False)
    df = merge_sources(df_weather, df_air)
    df = engineer_features(df, add_target_col=True)

    # Drop rows with NaN in features (start of series) or target (end of series)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def run_live_pipeline() -> pd.DataFrame:
    """
    Placeholder for the live twice-hourly pipeline.
    NOTE: current-endpoint calls only return a single latest reading, not a
    rolling window -- lag/rolling features need recent history to compute.
    This will be properly implemented once the Hopsworks Feature Group is in
    place, since at that point recent history will be READ BACK from Hopworks
    (not recomputed from scratch each run). Left as a TODO intentionally.
    """
    raise NotImplementedError(
        "Live pipeline will pull recent history from Hopworks Feature Store "
        "once the Feature Group schema is finalized -- see Day 4 plan."
    )


if __name__ == "__main__":
    # Quick manual test using a small date range
    df_test = build_training_dataset("2025-01-01", "2025-01-31")
    print(df_test.shape)
    print(df_test.head())