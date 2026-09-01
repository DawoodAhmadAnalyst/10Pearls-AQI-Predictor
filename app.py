import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta

from utils.hopsworks_utils import (
    connect_to_hopsworks,
    load_feature_data
)

from utils.prediction import predict_aqi, load_model_bundle, MODEL_NAMES
from utils.aqi_utils import get_aqi_category


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AQI Predictor | Multan",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN APP
    ===================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #F4F8FC 0%,
            #E8F1F8 50%,
            #F7FAFC 100%
        );
        color: #1F2937;
    }


    /* =====================================================
       HEADINGS
    ===================================================== */

    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }


    p, span, label {
        color: #374151;
    }


    /* =====================================================
       METRIC CARDS
    ===================================================== */

    div[data-testid="stMetric"] {

        background-color: #FFFFFF;

        border: 1px solid #DCE6F0;

        padding: 20px;

        border-radius: 16px;

        box-shadow:
            0px 4px 12px rgba(15, 23, 42, 0.08);

        min-height: 120px;
    }


    div[data-testid="stMetricLabel"] {

        color: #64748B !important;

        font-size: 14px !important;

        font-weight: 600 !important;
    }


    div[data-testid="stMetricValue"] {

        color: #0F172A !important;

        font-size: 28px !important;

        font-weight: 700 !important;
    }


    /* =====================================================
       EXPANDER
    ===================================================== */

    div[data-testid="stExpander"] {

        background-color: #FFFFFF;

        border: 1px solid #DCE6F0;

        border-radius: 12px;

        padding: 5px;
    }


    /* =====================================================
       DIVIDER
    ===================================================== */

    hr {
        border-color: #DCE6F0 !important;
    }


    /* =====================================================
       STREAMLIT ALERTS
    ===================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }


    /* =====================================================
       REMOVE DEFAULT STREAMLIT FOOTER
    ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HOPSWORKS CONNECTION
# ==========================================================

@st.cache_resource
def get_connections():

    project, feature_store, model_registry = connect_to_hopsworks()

    return feature_store, model_registry


# ==========================================================
# LOAD FEATURE DATA
# ==========================================================

@st.cache_data(ttl=3600)  # matches the hourly feature pipeline cadence
def get_data(_feature_store):

    return load_feature_data(_feature_store)


# ==========================================================
# AQI METRIC DISPLAY
# ==========================================================

def show_aqi_metric(column, title, value):

    info = get_aqi_category(value)

    with column:

        st.metric(
            label=title,
            value=f"{round(value)} AQI"
        )

        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 15px;
                font-weight: 700;
                color: {info['color']};
                margin-top: 5px;
            ">
                {info['emoji']} {info['category']}
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================================
# LOAD DATA
# ==========================================================

try:

    with st.spinner("Connecting to Hopsworks..."):

        feature_store, model_registry = get_connections()

        df = get_data(feature_store)


except Exception as e:

    st.error("❌ Unable to connect to Hopsworks.")

    st.exception(e)

    st.stop()


# ==========================================================
# PREPARE DATA
# ==========================================================

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)

latest_row = df.iloc[-1]          # Series — predict_aqi() reshapes this itself via .to_frame().T

current_aqi = latest_row["us_aqi"]


# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

try:

    with st.spinner("Generating AI predictions..."):

        # NOTE: predict_aqi() expects a Series here and reshapes it itself
        # via latest_row[feature_cols].to_frame().T — confirmed correct.
        # Still worth checking whether the model downloads inside this
        # function are cached (e.g. @st.cache_resource around
        # model_registry.get_model(...).download()), since Streamlit
        # reruns this whole script on every interaction and re-downloading
        # three model bundles from Hopsworks on every rerun would be slow
        # and burn through free-tier limits fast.
        predictions = predict_aqi(
            model_registry,
            latest_row
        )


except Exception as e:

    st.error("❌ Unable to generate AQI predictions.")

    st.exception(e)

    st.stop()


# ==========================================================
# HEADER
# ==========================================================

header_col1, header_col2 = st.columns([4, 1])


with header_col1:

    st.title("🌍 AQI Predictor")

    st.caption(
        "AI-Powered Air Quality Monitoring & Forecasting — Multan"
    )


with header_col2:

    st.success("🟢 STATION ACTIVE")

    st.caption(
        f"Last data update: {latest_row['time']}"
    )


st.divider()


# ==========================================================
# HAZARD ALERT BANNER
# ==========================================================

# US EPA: AQI > 150 crosses into "Unhealthy" territory. Checking current +
# all three forecasts (not just current) means a clean reading today doesn't
# hide a bad one arriving in the next 72h — and naming which horizon
# triggered it avoids needing a separate banner per reading.
UNHEALTHY_THRESHOLD = 150

all_readings = {
    "Current": current_aqi,
    "+24h": predictions["24h"],
    "+48h": predictions["48h"],
    "+72h": predictions["72h"],
}

worst_label, worst_value = max(all_readings.items(), key=lambda kv: kv[1])

if worst_value > UNHEALTHY_THRESHOLD:

    worst_info = get_aqi_category(worst_value)

    st.error(
        f"⚠️ **Hazard Alert — {worst_label} AQI forecast: "
        f"{round(worst_value)} ({worst_info['category']})**  \n"
        f"Air quality is expected to reach unhealthy levels. Limit outdoor "
        f"exposure, especially for children, the elderly, and those with "
        f"respiratory conditions."
    )


# ==========================================================
# AQI FORECAST
# ==========================================================

st.header("🌍 Air Quality Forecast")


col1, col2, col3, col4 = st.columns(4)


show_aqi_metric(
    col1,
    "CURRENT AQI",
    current_aqi
)


show_aqi_metric(
    col2,
    "FORECAST +24 HOURS",
    predictions["24h"]
)


show_aqi_metric(
    col3,
    "FORECAST +48 HOURS",
    predictions["48h"]
)


show_aqi_metric(
    col4,
    "FORECAST +72 HOURS",
    predictions["72h"]
)


# ==========================================================
# AQI TREND AND FORECAST
# ==========================================================

st.header("📈 AQI Trend & AI Forecast")


# Historical AQI
history_df = df.tail(72).copy()


fig = go.Figure()


# ----------------------------------------------------------
# HISTORICAL AQI
# ----------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=history_df["time"],
        y=history_df["us_aqi"],

        mode="lines",

        name="Historical AQI",

        line=dict(
            color="#0284C7",
            width=3
        )

    )

)


# ----------------------------------------------------------
# FORECAST DATA
# ----------------------------------------------------------

forecast_times = [

    latest_row["time"] + timedelta(hours=24),

    latest_row["time"] + timedelta(hours=48),

    latest_row["time"] + timedelta(hours=72)

]


forecast_values = [

    predictions["24h"],

    predictions["48h"],

    predictions["72h"]

]


# ----------------------------------------------------------
# AI FORECAST
# ----------------------------------------------------------

fig.add_trace(

    go.Scatter(

        x=[latest_row["time"]] + forecast_times,

        y=[current_aqi] + forecast_values,

        # Markers only, connected by a faint dotted guide — these are three
        # independent point-predictions from one feature vector, not a
        # continuous forecast curve, so the chart shouldn't imply a smooth
        # trajectory between them.
        mode="lines+markers",

        name="AI Forecast",

        line=dict(
            color="#F97316",
            width=1,
            dash="dot"
        ),

        marker=dict(
            size=11,
            symbol="diamond",
            color="#F97316",
            line=dict(width=1, color="#7C2D12")
        )

    )

)


# ----------------------------------------------------------
# CHART DESIGN
# ----------------------------------------------------------

fig.update_layout(

    height=450,

    template="plotly_white",

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="#FFFFFF",

    hovermode="x unified",

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),

    xaxis_title="Time",

    yaxis_title="AQI",

    font=dict(
        color="#1F2937"
    ),

    xaxis=dict(
        gridcolor="#E5E7EB"
    ),

    yaxis=dict(
        gridcolor="#E5E7EB"
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )

)


st.plotly_chart(
    fig,
    width="stretch"
)


st.divider()


# ==========================================================
# FEATURE IMPORTANCE (SHAP)
# ==========================================================

# Friendly display names for the features most likely to show up in the
# top-10 SHAP chart. Falls back to a generic underscore-to-title cleanup
# for anything not listed, so new/renamed features never crash the chart.
FEATURE_LABELS = {
    "aqi_roll_mean_24h": "24h Rolling Avg AQI",
    "aqi_roll_mean_6h": "6h Rolling Avg AQI",
    "aqi_roll_mean_12h": "12h Rolling Avg AQI",
    "aqi_lag_1h": "AQI (1h ago)",
    "aqi_lag_6h": "AQI (6h ago)",
    "aqi_lag_12h": "AQI (12h ago)",
    "aqi_lag_24h": "AQI (24h ago)",
    "pressure_lag_12h": "Pressure (12h ago)",
    "pressure_lag_24h": "Pressure (24h ago)",
    "temperature_2m": "Temperature",
    "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "Wind Speed",
    "surface_pressure": "Pressure",
    "cloud_cover": "Cloud Cover",
    "precipitation": "Precipitation",
    "us_aqi": "Current AQI",
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
    "dust": "Dust",
}


def prettify_feature_name(name):
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_", " ").title()


st.header("🔍 Why This Forecast?")

st.caption(
    "SHAP values show how much each input pushed the selected forecast "
    "above or below the model's average prediction."
)

shap_horizon = st.selectbox(
    "Explain forecast for:",
    options=["24h", "48h", "72h"],
    format_func=lambda h: f"+{h}"
)

try:

    with st.spinner("Computing SHAP values..."):

        # load_model_bundle is cached (see utils/prediction.py), so this
        # doesn't trigger another Hopsworks download beyond the one
        # predict_aqi() already made for this same horizon.
        shap_bundle = load_model_bundle(
            model_registry,
            MODEL_NAMES[shap_horizon]
        )

        shap_model = shap_bundle["model"]
        shap_scaler = shap_bundle["scaler"]
        shap_feature_cols = shap_bundle["feature_cols"]

        # Background sample for the explainer's baseline — recent
        # historical rows, scaled the same way the model was trained on.
        background_raw = df[shap_feature_cols].sample(
            n=min(100, len(df)),
            random_state=42
        )
        background_scaled = shap_scaler.transform(background_raw)

        # Ridge is a linear model, so LinearExplainer gives exact SHAP
        # values in closed form — no sampling/approximation needed, unlike
        # tree-based (TreeExplainer) or arbitrary (KernelExplainer) models.
        explainer = shap.LinearExplainer(shap_model, background_scaled)

        X_instance = latest_row[shap_feature_cols].to_frame().T
        X_instance_scaled = shap_scaler.transform(X_instance)

        shap_values = explainer.shap_values(X_instance_scaled)[0]

    shap_df = pd.DataFrame({
        "Feature": shap_feature_cols,
        "SHAP Value": shap_values
    })

    shap_df["Direction"] = shap_df["SHAP Value"].apply(
        lambda v: "Increases AQI" if v > 0 else "Decreases AQI"
    )

    top_features = shap_df.reindex(
        shap_df["SHAP Value"].abs().sort_values(ascending=False).index
    ).head(10).copy()

    top_features["Feature"] = top_features["Feature"].apply(prettify_feature_name)

    shap_fig = px.bar(
        top_features.sort_values("SHAP Value"),
        x="SHAP Value",
        y="Feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Increases AQI": "#F97316",
            "Decreases AQI": "#0284C7"
        },
        template="plotly_white"
    )

    shap_fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1F2937"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title=f"Impact on +{shap_horizon} AQI prediction", gridcolor="#E5E7EB"),
        yaxis=dict(title="", gridcolor="#E5E7EB"),
        legend_title_text=""
    )

    st.plotly_chart(
        shap_fig,
        width="stretch"
    )

    st.caption(
        "Note: several AQI-related inputs (rolling averages and lags) are "
        "highly correlated with each other. Ridge can split their combined "
        "signal across them with offsetting signs — so an individual "
        "feature's direction here reflects its adjustment *relative to the "
        "others*, not its effect in isolation."
    )

except Exception as e:

    st.warning("Couldn't compute SHAP values for this forecast.")
    st.exception(e)


# ==========================================================
# POLLUTANTS AND WEATHER
# ==========================================================

left_col, right_col = st.columns(2)


# ==========================================================
# POLLUTANT LEVELS
# ==========================================================

with left_col:

    st.header("🏭 Current Pollutant Levels")


    pollutant_data = {

        "Pollutant": [

            "PM2.5",
            "PM10",
            "Carbon Monoxide",
            "Nitrogen Dioxide",
            "Sulphur Dioxide",
            "Ozone",
            "Dust"

        ],

        "Value": [

            latest_row.get("pm2_5", 0),
            latest_row.get("pm10", 0),
            latest_row.get("carbon_monoxide", 0),
            latest_row.get("nitrogen_dioxide", 0),
            latest_row.get("sulphur_dioxide", 0),
            latest_row.get("ozone", 0),
            latest_row.get("dust", 0)

        ]

    }


    pollutant_df = pd.DataFrame(pollutant_data)


    # Note: pollutants are in different units and magnitudes (CO in the
    # hundreds/thousands µg/m³, SO2 in the tens, etc.), so coloring bars by
    # raw value on one shared scale would be redundant with bar length at
    # best and misleading at worst. Bar length alone carries the value here.
    pollutant_fig = px.bar(

        pollutant_df,

        x="Value",

        y="Pollutant",

        orientation="h",

        template="plotly_white",

        text="Value",

        color_discrete_sequence=["#0284C7"]

    )


    pollutant_fig.update_layout(

        height=420,

        paper_bgcolor="rgba(0,0,0,0)",

        plot_bgcolor="#FFFFFF",

        font=dict(
            color="#1F2937"
        ),

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        coloraxis_showscale=False,

        xaxis=dict(
            gridcolor="#E5E7EB"
        ),

        yaxis=dict(
            gridcolor="#E5E7EB"
        )

    )


    st.plotly_chart(

        pollutant_fig,

        width="stretch"

    )


# ==========================================================
# WEATHER CONDITIONS
# ==========================================================

with right_col:

    st.header("🌤️ Current Weather Conditions")


    weather_col1, weather_col2 = st.columns(2)


    with weather_col1:

        st.metric(

            "🌡️ Temperature",

            f"{latest_row.get('temperature_2m', 0):.1f} °C"

        )


        st.metric(

            "💨 Wind Speed",

            f"{latest_row.get('wind_speed_10m', 0):.1f} km/h"

        )


        st.metric(

            "☁️ Cloud Cover",

            f"{latest_row.get('cloud_cover', 0):.0f} %"

        )


    with weather_col2:

        st.metric(

            "💧 Humidity",

            f"{latest_row.get('relative_humidity_2m', 0):.0f} %"

        )


        st.metric(

            "⚡ Pressure",

            f"{latest_row.get('surface_pressure', 0):.1f} hPa"

        )


        st.metric(

            "🌧️ Precipitation",

            f"{latest_row.get('precipitation', 0):.1f} mm"

        )


# ==========================================================
# MACHINE LEARNING SYSTEM
# ==========================================================

st.header("🤖 Machine Learning System")


ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)


with ml_col1:

    st.metric(
        "Model",
        "Ridge Regression"
    )


with ml_col2:

    st.metric(
        "Feature Store",
        "Hopsworks"
    )


with ml_col3:

    st.metric(
        "Feature Update",
        "Hourly"
    )


with ml_col4:

    st.metric(
        "Model Update",
        "Daily"
    )


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

with st.expander("🔍 View Latest Feature Record"):

    st.dataframe(
        latest_row.to_frame().T,
        width="stretch"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()


st.caption(
    "🌍 AQI Predictor • AI-Powered Air Quality Forecasting • "
    "Powered by Hopsworks Feature Store & Model Registry"
)