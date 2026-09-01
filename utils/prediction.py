import os
import joblib
import tempfile
import streamlit as st


MODEL_NAMES = {
    "24h": "aqi_ridge_24h",
    "48h": "aqi_ridge_48h",
    "72h": "aqi_ridge_72h"
}


@st.cache_resource(show_spinner="Loading model bundle from Hopsworks...")
def load_model_bundle(_model_registry, model_name):
    """
    Load the latest version of a model from Hopsworks.
    Cached per model_name — without this, Streamlit's full-script rerun on
    every interaction would re-download all three model bundles from
    Hopsworks every time (slow, and burns free-tier request limits).
    The leading underscore on _model_registry tells st.cache_resource not to
    try to hash that argument (it isn't hashable); caching keys off
    model_name instead, which is exactly what should invalidate the cache.

    Returns:
        Dictionary containing:
        - model
        - scaler
        - feature_cols
    """

    model = _model_registry.get_model(
        name=model_name
    )

    model_path = model.download()

    # Find the PKL file
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".pkl"):
                bundle_path = os.path.join(root, file)
                return joblib.load(bundle_path)

    raise FileNotFoundError(
        f"No .pkl model file found for {model_name}"
    )


def predict_aqi(model_registry, latest_row):
    """
    Generate AQI predictions for:

        +24 hours
        +48 hours
        +72 hours
    """

    predictions = {}

    for horizon, model_name in MODEL_NAMES.items():

        bundle = load_model_bundle(
            model_registry,
            model_name
        )

        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_cols = bundle["feature_cols"]

        # Extract features in EXACT training order
        X = latest_row[feature_cols].to_frame().T

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]

        predictions[horizon] = round(float(prediction), 1)

    return predictions