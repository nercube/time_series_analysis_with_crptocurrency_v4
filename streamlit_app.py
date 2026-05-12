import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR.parent / "models"
DATA_DIR = BASE_DIR.parent / "data"

st.set_page_config(layout="wide")

CRYPTO_CONFIG = {

    "Bitcoin (BTC)": {

        "data": "btc.csv",

        "arima": "arima_model.pkl",

        "prophet": "prophet_model.pkl",

        "lstm": "btc_lstm_logreturn_multifeature.h5",

        "scaler": "btc_feature_scaler.pkl",
    },

    "Ethereum (ETH)": {

        "data": "eth.csv",

        "arima": "eth_arima.pkl",

        "prophet": "eth_prophet (1).pkl",

        "lstm": "eth_lstm (1).h5",

        "scaler": "eth_scaler (1).pkl",
    },

    "Tether (USDT)": {

        "data": "usdt.csv",

        "arima": "usdt_arima.pkl",

        "prophet": "usdt_prophet (1).pkl",

        "lstm": "usdt_lstm (1).h5",

        "scaler": "usdt_scaler.pkl",
    },
}

selected_crypto = st.sidebar.radio(
    "Choose Cryptocurrency",
    list(CRYPTO_CONFIG.keys())
)

cfg = CRYPTO_CONFIG[selected_crypto]

st.title(f"{selected_crypto} Forecast Dashboard")

# =========================
# LOAD CSV
# =========================

csv_path = DATA_DIR / cfg["data"]

df = pd.read_csv(csv_path)

st.subheader("CSV Loaded")

st.write(df.head())

# =========================
# LOAD MODELS
# =========================

st.subheader("Loading Models")

try:

    arima_model = joblib.load(
        MODELS_DIR / cfg["arima"]
    )

    st.success("ARIMA loaded")

except Exception as e:

    st.error(f"ARIMA ERROR: {e}")

try:

    prophet_model = joblib.load(
        MODELS_DIR / cfg["prophet"]
    )

    st.success("Prophet loaded")

except Exception as e:

    st.error(f"PROPHET ERROR: {e}")

try:

    lstm_model = load_model(
        MODELS_DIR / cfg["lstm"],
        compile=False
    )

    st.success("LSTM loaded")

except Exception as e:

    st.error(f"LSTM ERROR: {e}")

try:

    scaler = joblib.load(
        MODELS_DIR / cfg["scaler"]
    )

    st.success("Scaler loaded")

except Exception as e:

    st.error(f"SCALER ERROR: {e}")
