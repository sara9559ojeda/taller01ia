"""Shared preprocessing for the LSTM energy-demand workshop.

Used by both the modeling notebook and the Streamlit app so that training and
serving apply exactly the same feature engineering and scaling.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CLEAN_DATA_PATH = Path("data/energy_demand_clean.csv")
MODELS_DIR = Path("models")
SCALERS_PATH = MODELS_DIR / "scalers.joblib"
CONFIG_PATH = MODELS_DIR / "pipeline_config.json"

TARGET_COL = "demanda_objetivo"

# Continuous variables observed up to hour t (standardized with train statistics).
CONTINUOUS_COLS = [
    "demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh",
    "radiacion_wm2", "precipitacion_mm", "precio_kwh",
]
# Binary calendar flags (already 0/1, not scaled).
BINARY_COLS = ["fin_semana", "festivo"]
# Cyclical encodings of hour, weekday and month (already in [-1, 1], not scaled).
CYCLICAL_COLS = ["hora_sin", "hora_cos", "dia_sin", "dia_cos", "mes_sin", "mes_cos"]

FEATURE_COLS = CONTINUOUS_COLS + BINARY_COLS + CYCLICAL_COLS

# Chronological split, assigned by the timestamp of the predicted hour (t + 1).
VAL_START = pd.Timestamp("2026-07-01 00:00:00")
TEST_START = pd.Timestamp("2026-10-01 00:00:00")


def load_clean_data(path=CLEAN_DATA_PATH):
    frame = pd.read_csv(path, parse_dates=["timestamp"], encoding="utf-8")
    return frame.sort_values("timestamp").reset_index(drop=True)


def add_calendar_features(frame):
    """Derives calendar columns from the timestamp plus sin/cos encodings.

    The sin/cos pair places 23:00 next to 00:00 (and December next to January),
    which a raw integer encoding cannot express.
    """
    frame = frame.copy()
    timestamps = frame["timestamp"]
    frame["hora"] = timestamps.dt.hour
    frame["dia_semana"] = timestamps.dt.dayofweek
    frame["mes"] = timestamps.dt.month
    frame["fin_semana"] = (frame["dia_semana"] >= 5).astype(int)
    if "festivo" not in frame:
        frame["festivo"] = 0
    frame["hora_sin"] = np.sin(2 * np.pi * frame["hora"] / 24)
    frame["hora_cos"] = np.cos(2 * np.pi * frame["hora"] / 24)
    frame["dia_sin"] = np.sin(2 * np.pi * frame["dia_semana"] / 7)
    frame["dia_cos"] = np.cos(2 * np.pi * frame["dia_semana"] / 7)
    frame["mes_sin"] = np.sin(2 * np.pi * (frame["mes"] - 1) / 12)
    frame["mes_cos"] = np.cos(2 * np.pi * (frame["mes"] - 1) / 12)
    return frame


def scale_features(frame, feature_scaler):
    """Returns the model input matrix (rows x FEATURE_COLS) as float32."""
    scaled = frame[FEATURE_COLS].astype("float64").copy()
    scaled[CONTINUOUS_COLS] = feature_scaler.transform(frame[CONTINUOUS_COLS])
    return scaled.to_numpy(dtype="float32")


def build_sequences(feature_matrix, targets, target_times, window):
    """Sliding windows: inputs are hours t-window+1..t, target is demand at t+1.

    Only past and present rows enter each window, so no future information is
    used as input. Samples whose target is missing are dropped.
    """
    windows = np.lib.stride_tricks.sliding_window_view(feature_matrix, window, axis=0)
    windows = np.transpose(windows, (0, 2, 1))  # (samples, window, features)
    last_row = np.arange(window - 1, len(feature_matrix))
    sample_targets = targets[last_row]
    sample_times = target_times[last_row]
    valid = ~np.isnan(sample_targets)
    return windows[valid], sample_targets[valid], sample_times[valid]


def split_masks(sample_times):
    sample_times = pd.DatetimeIndex(sample_times)
    train = sample_times < VAL_START
    val = (sample_times >= VAL_START) & (sample_times < TEST_START)
    test = sample_times >= TEST_START
    return train, val, test


def save_preprocessing(feature_scaler, target_scaler, extra_config=None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, SCALERS_PATH)
    config = {
        "feature_cols": FEATURE_COLS,
        "continuous_cols": CONTINUOUS_COLS,
        "val_start": str(VAL_START),
        "test_start": str(TEST_START),
    }
    config.update(extra_config or {})
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def load_preprocessing():
    scalers = joblib.load(SCALERS_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return scalers["feature_scaler"], scalers["target_scaler"], config


def prepare_window(window_frame, feature_scaler):
    """Turns a frame with `window` consecutive hourly rows into a model batch of size 1."""
    enriched = add_calendar_features(window_frame)
    return scale_features(enriched, feature_scaler)[np.newaxis, ...]
