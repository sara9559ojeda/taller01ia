"""Streamlit app: next-hour electricity demand forecast with the trained LSTM models.

Run with:  streamlit run app.py
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path

import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lstm_pipeline as pipeline

RESULTS_PATH = Path("outputs/reports/results_table_test.csv")
FIGURES_DIR = Path("outputs/figures")
INPUT_COLS = ["demanda_mw", "temperatura_c", "humedad_pct", "viento_kmh", "radiacion_wm2",
              "precipitacion_mm", "precio_kwh", "festivo"]
# column -> (label, min, max, step)
FIELD_SPECS = {
    "demanda_mw": ("Demanda actual (MW)", 0.0, 3000.0, 1.0),
    "temperatura_c": ("Temperatura (°C)", -10.0, 45.0, 0.1),
    "humedad_pct": ("Humedad relativa (%)", 0.0, 100.0, 0.5),
    "viento_kmh": ("Viento (km/h)", 0.0, 150.0, 0.5),
    "radiacion_wm2": ("Radiación solar (W/m²)", 0.0, 1400.0, 5.0),
    "precipitacion_mm": ("Precipitación (mm)", 0.0, 100.0, 0.1),
    "precio_kwh": ("Precio de la energía (por kWh)", 0.01, 200.0, 0.1),
}
RANGE_RULES = {
    "demanda_mw": (0.0, None, "debe ser mayor que 0"),
    "humedad_pct": (0.0, 100.0, "debe estar entre 0 y 100"),
    "viento_kmh": (0.0, None, "no puede ser negativo"),
    "radiacion_wm2": (0.0, None, "no puede ser negativa"),
    "precipitacion_mm": (0.0, None, "no puede ser negativa"),
    "precio_kwh": (0.0, None, "debe ser mayor que 0"),
}
WEEKDAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
COLOR_HISTORY = "#2a78d6"
COLOR_PREDICTION = "#eb6834"
COLOR_ACTUAL = "#0b0b0b"

st.set_page_config(page_title="Demanda eléctrica · LSTM", page_icon="⚡", layout="wide")


@st.cache_resource(show_spinner=False)
def load_preprocessing():
    return pipeline.load_preprocessing()


@st.cache_resource(show_spinner="Cargando modelo…")
def load_model(file_name):
    return keras.models.load_model(pipeline.MODELS_DIR / file_name)


@st.cache_data(show_spinner=False)
def load_history():
    return pipeline.load_clean_data()


@st.cache_data(show_spinner=False)
def load_results():
    return pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else None


def format_time_es(timestamp):
    return f"{WEEKDAYS_ES[timestamp.dayofweek]} {timestamp:%d/%m/%Y %H:%M}"


def find_range_problems(frame):
    problems = []
    if frame[INPUT_COLS].isna().any().any():
        problems.append("Hay celdas vacías en la ventana.")
    for col, (low, high, message) in RANGE_RULES.items():
        values = frame[col]
        out_of_range = values <= low if col in ("demanda_mw", "precio_kwh") else values < low
        if high is not None:
            out_of_range |= values > high
        if out_of_range.any():
            problems.append(f"`{col}` {message} ({int(out_of_range.sum())} hora(s)).")
    if not frame["festivo"].isin([0, 1]).all():
        problems.append("`festivo` solo admite 0 o 1.")
    return problems


def read_uploaded_window(uploaded_file, window):
    frame = pd.read_csv(uploaded_file)
    missing = [col for col in ["timestamp"] + INPUT_COLS if col not in frame.columns]
    if missing:
        return None, f"Faltan columnas en el CSV: {', '.join(missing)}"
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        return None, "Hay valores de `timestamp` que no son fechas válidas (formato esperado: AAAA-MM-DD HH:MM:SS)."
    frame = frame.sort_values("timestamp").tail(window).reset_index(drop=True)
    if len(frame) < window:
        return None, f"El modelo seleccionado necesita {window} horas y el CSV tiene {len(frame)}."
    if (frame["timestamp"].diff().dropna() != pd.Timedelta(hours=1)).any():
        return None, "Las horas del CSV deben ser consecutivas (una fila por hora, sin saltos)."
    return frame[["timestamp"] + INPUT_COLS], None


def predict_next_hour(model, input_frame, feature_scaler, target_scaler):
    batch = pipeline.prepare_window(input_frame, feature_scaler)
    scaled_prediction = model(batch, training=False).numpy()
    return float(target_scaler.inverse_transform(scaled_prediction)[0, 0])


# ---------------------------------------------------------------- artifacts
if not pipeline.CONFIG_PATH.exists() or not pipeline.SCALERS_PATH.exists():
    st.error("No se encontraron los modelos entrenados. Ejecuta primero `01_data_cleaning.ipynb` "
             "y `02_lstm_modeling.ipynb`.")
    st.stop()

feature_scaler, target_scaler, config = load_preprocessing()
history = load_history()
results = load_results()
best_model_info = config["best_model"]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Modelo")
model_options = config["models"]
best_index = next(i for i, option in enumerate(model_options) if option["file"] == best_model_info["file"])


def describe_option(option):
    text = f"{option['label']} · ventana {option['window']} h"
    return f"{text}  ⭐" if option["file"] == best_model_info["file"] else text


selected = st.sidebar.selectbox("Arquitectura y ventana", model_options, index=best_index, format_func=describe_option)
window = int(selected["window"])
model = load_model(selected["file"])
st.sidebar.caption("⭐ = mejor modelo según el error en validación.")

if results is not None:
    selected_metrics = results[(results["Modelo"] == selected["label"]) & (results["Ventana"] == window)]
    if not selected_metrics.empty:
        metrics_row = selected_metrics.iloc[0]
        st.sidebar.subheader("Desempeño en prueba")
        st.sidebar.caption("Oct–dic 2026, datos nunca vistos en el entrenamiento")
        left, right = st.sidebar.columns(2)
        left.metric("MAE (MW)", f"{metrics_row['MAE']:.1f}")
        right.metric("RMSE (MW)", f"{metrics_row['RMSE']:.1f}")
        left.metric("MAPE (%)", f"{metrics_row['MAPE (%)']:.2f}")
        right.metric("R²", f"{metrics_row['R²']:.3f}")

# ---------------------------------------------------------------- main
st.title("⚡ Predicción de la demanda eléctrica de la próxima hora")
st.markdown(
    f"El modelo recibe las **variables independientes (X)** de las últimas **{window} horas** "
    f"(demanda, clima, precio y calendario) y predice la **demanda de la hora siguiente (y)**."
)
prediction_tab, comparison_tab, about_tab = st.tabs(["🔮 Predicción", "📊 Comparación de modelos", "ℹ️ Cómo funciona"])

with prediction_tab:
    st.subheader("1. Momento de la predicción")
    first_allowed = history["timestamp"].iloc[window - 1]
    last_allowed = history["timestamp"].iloc[-1]
    date_col, hour_col, info_col = st.columns([1.2, 0.8, 2])
    chosen_date = date_col.date_input("Fecha de la hora actual (t)", value=last_allowed.date(),
                                      min_value=first_allowed.date(), max_value=last_allowed.date(), format="DD/MM/YYYY")
    chosen_hour = hour_col.selectbox("Hora actual (t)", list(range(24)), index=int(last_allowed.hour),
                                     format_func=lambda hour: f"{hour:02d}:00")
    current_time = max(pd.Timestamp(chosen_date) + pd.Timedelta(hours=chosen_hour), first_allowed)
    target_time = current_time + pd.Timedelta(hours=1)
    info_col.info(f"Se predice la demanda de **{format_time_es(target_time)}** "
                  f"usando las horas {current_time - pd.Timedelta(hours=window - 1):%d/%m %H:%M} → {current_time:%d/%m %H:%M}.")

    in_window = (history["timestamp"] > current_time - pd.Timedelta(hours=window)) & (history["timestamp"] <= current_time)
    original_window = history.loc[in_window, ["timestamp"] + INPUT_COLS].reset_index(drop=True)
    key_suffix = f"{current_time:%Y%m%d%H}_{window}"

    st.subheader("2. Variables independientes (X)")
    source = st.radio("Origen de los datos de entrada", ["Formulario (histórico + mis ajustes)", "Cargar un CSV propio"],
                      horizontal=True, label_visibility="collapsed")

    if source.startswith("Formulario"):
        st.markdown(f"**Hora actual t = {format_time_es(current_time)}** · ajusta los valores para simular un escenario:")
        current_row = original_window.iloc[-1]
        field_cols = st.columns(4)
        user_values = {}
        for position, (col, (label, low, high, step)) in enumerate(FIELD_SPECS.items()):
            default_value = min(max(float(current_row[col]), low), high)
            user_values[col] = field_cols[position % 4].number_input(label, min_value=low, max_value=high,
                                                                      value=default_value, step=step,
                                                                      key=f"{col}_{key_suffix}")
        user_values["festivo"] = int(field_cols[3].checkbox("Día festivo", value=bool(current_row["festivo"]),
                                                            key=f"festivo_{key_suffix}"))
        st.caption(f"Calendario derivado automáticamente de la fecha: {WEEKDAYS_ES[current_time.dayofweek]}, "
                   f"hora {current_time.hour}, mes {current_time.month}, "
                   f"fin de semana: {'sí' if current_time.dayofweek >= 5 else 'no'}.")

        with st.expander(f"Ver o editar las {window - 1} horas anteriores (t−{window - 1} … t−1)"):
            edited_previous = st.data_editor(original_window.iloc[:-1], key=f"editor_{key_suffix}", hide_index=True,
                                             disabled=["timestamp"], width="stretch",
                                             column_config={"timestamp": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm")})
        temperature_shift = st.slider("Escenario climático: sumar a la temperatura de toda la ventana (°C)",
                                      -5.0, 5.0, 0.0, 0.5, key=f"shift_{key_suffix}")
        input_frame = pd.concat([edited_previous, pd.DataFrame([{"timestamp": current_time, **user_values}])],
                                ignore_index=True)
        input_frame["temperatura_c"] = input_frame["temperatura_c"] + temperature_shift
    else:
        st.markdown(f"El CSV debe tener **{window} filas horarias consecutivas** y las columnas "
                    f"`timestamp, {', '.join(INPUT_COLS)}`. Se predice la hora siguiente a la última fila.")
        st.download_button("Descargar plantilla (ventana actual)", original_window.to_csv(index=False).encode("utf-8"),
                           file_name=f"ventana_{window}h.csv", mime="text/csv")
        uploaded = st.file_uploader("CSV de entrada", type="csv")
        if uploaded is None:
            st.stop()
        input_frame, upload_error = read_uploaded_window(uploaded, window)
        if upload_error:
            st.error(upload_error)
            st.stop()
        current_time = input_frame["timestamp"].iloc[-1]
        target_time = current_time + pd.Timedelta(hours=1)

    problems = find_range_problems(input_frame)
    if problems:
        st.error("Revisa los datos de entrada:\n\n" + "\n".join(f"- {problem}" for problem in problems))
        st.stop()

    st.subheader("3. Predicción (y)")
    prediction = predict_next_hour(model, input_frame, feature_scaler, target_scaler)
    actual_values = history.loc[history["timestamp"] == target_time, "demanda_mw"]
    reference_window = history.set_index("timestamp").reindex(input_frame["timestamp"])[INPUT_COLS]
    inputs_modified = reference_window.isna().any().any() or not np.allclose(
        input_frame[INPUT_COLS].to_numpy(dtype=float), reference_window.to_numpy(dtype=float))
    current_demand = float(input_frame["demanda_mw"].iloc[-1])

    result_cols = st.columns(3)
    result_cols[0].metric(f"Demanda predicha · {target_time:%d/%m/%Y %H:%M}", f"{prediction:,.1f} MW",
                          delta=f"{prediction - current_demand:+.1f} MW respecto a la hora actual", delta_color="off")
    if not actual_values.empty:
        actual = float(actual_values.iloc[0])
        result_cols[1].metric("Demanda real registrada", f"{actual:,.1f} MW")
        result_cols[2].metric("Error de la predicción", f"{abs(prediction - actual):,.1f} MW",
                              delta=f"{abs(prediction - actual) / actual * 100:.2f} % del valor real", delta_color="off")
        if inputs_modified:
            st.caption("Modificaste las entradas: la demanda real corresponde al escenario original, "
                       "así que el error mide cuánto se aleja tu escenario de lo que ocurrió.")
    else:
        result_cols[1].metric("Demanda real registrada", "—")
        result_cols[1].caption("Hora fuera del dataset: es un pronóstico genuino.")

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=input_frame["timestamp"], y=input_frame["demanda_mw"], mode="lines+markers",
                                name=f"demanda de entrada ({window} h)", line={"color": COLOR_HISTORY, "width": 2},
                                marker={"size": 5}))
    figure.add_trace(go.Scatter(x=[current_time, target_time], y=[current_demand, prediction], mode="lines",
                                line={"color": COLOR_PREDICTION, "width": 2, "dash": "dash"}, showlegend=False,
                                hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=[target_time], y=[prediction], mode="markers", name="predicción",
                                marker={"color": COLOR_PREDICTION, "size": 13, "symbol": "diamond"}))
    if not actual_values.empty:
        figure.add_trace(go.Scatter(x=[target_time], y=[float(actual_values.iloc[0])], mode="markers", name="real",
                                    marker={"color": COLOR_ACTUAL, "size": 10, "symbol": "circle-open",
                                            "line": {"width": 2}}))
    figure.update_layout(height=380, margin={"l": 10, "r": 10, "t": 30, "b": 10}, yaxis_title="MW",
                         legend={"orientation": "h", "y": 1.12}, hovermode="x unified")
    st.plotly_chart(figure, width="stretch")

    with st.expander("Ver la ventana exacta que recibe el modelo"):
        st.dataframe(input_frame, hide_index=True, width="stretch")

with comparison_tab:
    if results is None:
        st.info("Ejecuta `02_lstm_modeling.ipynb` para generar la tabla de resultados.")
    else:
        st.subheader("Tabla de resultados (conjunto de prueba)")
        st.dataframe(results.style.format({"MAE": "{:.2f}", "MSE": "{:.1f}", "RMSE": "{:.2f}", "MAPE (%)": "{:.2f}",
                                           "R²": "{:.4f}", "Parámetros": "{:,}", "Tiempo (s)": "{:.0f}"})
                     .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="#cde2fb")
                     .highlight_max(subset=["R²"], color="#cde2fb"),
                     hide_index=True, width="stretch")
    for file_name, caption in [("model_comparison.png", "Comparación de modelos"),
                               ("real_vs_predicted_best.png", "Demanda real vs. predicha (mejor modelo)"),
                               ("prediction_error_best.png", "Error de predicción (mejor modelo)"),
                               ("loss_curves.png", "Loss de entrenamiento y validación")]:
        if (FIGURES_DIR / file_name).exists():
            st.image(str(FIGURES_DIR / file_name), caption=caption, width="stretch")

with about_tab:
    st.markdown(f"""
**Datos.** Serie horaria 2025–2026 limpiada en `01_data_cleaning.ipynb`: sin duplicados, ordenada, anomalías marcadas
y tratadas sin inventar valores.

**Variables de entrada (X), por cada hora de la ventana:** demanda, temperatura, humedad, viento, radiación,
precipitación, precio, fin de semana, festivo, y hora / día de la semana / mes codificados con seno y coseno.

**Target (y):** demanda de la hora siguiente (`demanda_objetivo`). Nunca se usa como entrada.

**Preprocesamiento.** La normalización usa la media y desviación del periodo de **entrenamiento** (ene 2025 – jun 2026);
la app aplica exactamente la misma transformación (`lstm_pipeline.py`).

**Modelos.** Tres arquitecturas construidas solo con capas LSTM, Dense y Dropout (A. LSTM base, B. LSTM profunda y
C. LSTM con mayor capacidad + Dropout) × tres ventanas (12, 24 y 48 h), todas entrenadas con Early Stopping.
El recomendado (⭐) es **{best_model_info['label']} con ventana de {best_model_info['window']} h**, elegido por su error en validación.

**Limitaciones.** Predice solo una hora hacia adelante; los escenarios muy alejados de lo observado en 2025–2026
(por ejemplo temperaturas extremas) son extrapolaciones y deben interpretarse con cautela.
""")
