import os
import tempfile
import pandas as pd
import hopsworks
import streamlit as st
from dotenv import load_dotenv


# Load environment variables (only present locally — Streamlit Community
# Cloud has no .env file, so this is a no-op there and that's expected)
load_dotenv()


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

FEATURE_GROUP_NAME = "aqi_features_multan"
FEATURE_GROUP_VERSION = 2


# -------------------------------------------------------------------
# Hopsworks Connection
# -------------------------------------------------------------------

def connect_to_hopsworks():
    """
    Connect to Hopsworks and return:

        project
        feature_store
        model_registry
    """

    # Local dev reads from .env via os.getenv. Streamlit Community Cloud
    # injects secrets into st.secrets instead, not into os.environ — so
    # check both rather than assuming one deployment environment. Wrapped
    # in try/except because st.secrets can raise if no secrets.toml exists
    # at all (e.g. pure local dev with only a .env file).
    api_key = os.getenv("AQI_Predictor_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("AQI_Predictor_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "Hopsworks API key not found. Set AQI_Predictor_KEY in your "
            "local .env file, or in this app's Secrets if running on "
            "Streamlit Community Cloud."
        )

    # Was hardcoded to D:\tmp (Windows-only). Streamlit Community Cloud runs
    # Linux containers, so that path doesn't exist there — use the OS's own
    # temp directory instead, which works on both.
    os.makedirs(os.path.join(tempfile.gettempdir(), "hopsworks_tmp"), exist_ok=True)

    project = hopsworks.login(
        api_key_value=api_key,
        cert_folder="./hopsworks-certs"
    )

    feature_store = project.get_feature_store()
    model_registry = project.get_model_registry()

    return project, feature_store, model_registry


# -------------------------------------------------------------------
# Feature Group
# -------------------------------------------------------------------

def get_feature_group(feature_store):
    """Get the AQI Feature Group."""

    return feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )


def load_feature_data(feature_store):
    """
    Load all Feature Group data and sort chronologically.
    """

    feature_group = get_feature_group(feature_store)

    df = feature_group.read()

    df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    return df


def get_latest_data(feature_store):
    """
    Return the most recent valid row from the Feature Group.
    """

    df = load_feature_data(feature_store)

    latest_row = df.iloc[-1].copy()

    return latest_row


def get_recent_data(feature_store, hours=72):
    """
    Return recent historical data for charts.
    """

    df = load_feature_data(feature_store)

    df = df.tail(hours).copy()

    return df