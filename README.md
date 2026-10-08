# Taller LSTM – Predicción de demanda energética

Predicción de la demanda eléctrica de la **hora siguiente** a partir de una ventana de *n* horas (12, 24 o 48) con información histórica multivariada.

## Estructura

| Archivo / carpeta | Contenido |
|---|---|
| `dataset_demanda_energia_LSTM_2025_2026.csv` | Dataset original (sin modificar) |
| `01_data_cleaning.ipynb` | **Parte 1** – diagnóstico y limpieza, con la justificación de cada regla |
| `02_lstm_modeling.ipynb` | **Parte 2** – features, split cronológico, normalización, 3 arquitecturas × 3 ventanas, métricas, gráficas, sobreajuste, experimentos y respuestas a las preguntas |
| `lstm_pipeline.py` | Preprocesamiento compartido por el notebook y la app (garantiza que entrenamiento y despliegue transformen igual los datos) |
| `app.py` | **Despliegue** – app web Streamlit: se ingresan las X y se obtiene la predicción y |
| `data/energy_demand_clean.csv` | Dataset limpio |
| `models/` | Modelos entrenados (`.keras`), scalers y configuración |
| `outputs/figures/` | Todas las gráficas |
| `outputs/reports/` | Tabla de resultados, registro de limpieza (resumen y cambio por cambio), experimentos |

## Cómo ejecutar

```bash
pip install -r requirements.txt
```

Ejecutar los notebooks en orden (`01` y luego `02`) desde esta carpeta. En `02_lstm_modeling.ipynb`, `RETRAIN = False` reutiliza los modelos ya guardados en `models/`; con `True` se reentrena todo (≈ 25 min en CPU).

Para abrir la app web:

```bash
streamlit run app.py
```

## Arquitecturas (solo LSTM + Dense + Dropout, con Early Stopping)

| | Estructura | Parámetros |
|---|---|---|
| A. LSTM base | `LSTM(64) → Dense(1)` | 20 545 |
| B. LSTM profunda | `LSTM(64, return_sequences=True) → LSTM(32) → Dense(1)` | 32 929 |
| C. Mayor capacidad + regularización | `LSTM(128, return_sequences=True) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32, relu) → Dense(1)` | 125 249 |

## Resultados (conjunto de prueba: oct–dic 2026)

| Modelo | Ventana | MAE | RMSE | MAPE | R² | Épocas |
|---|---|---|---|---|---|---|
| A. LSTM base | 12 h | 19.84 | 24.83 | 3.39 % | 0.903 | 25 |
| A. LSTM base | 24 h | 19.75 | 24.81 | 3.38 % | 0.903 | 23 |
| A. LSTM base | 48 h | 20.21 | 25.35 | 3.47 % | 0.899 | 32 |
| B. LSTM profunda | 12 h | 20.10 | 25.24 | 3.44 % | 0.900 | 20 |
| **B. LSTM profunda** | **24 h** | **19.59** | **24.68** | **3.36 %** | **0.904** | 24 |
| B. LSTM profunda | 48 h | 20.57 | 26.22 | 3.54 % | 0.892 | 17 |
| C. Capacidad + Dropout | 12 h | 19.63 | 24.77 | 3.37 % | 0.904 | 27 |
| C. Capacidad + Dropout | 24 h | 20.17 | 25.21 | 3.45 % | 0.900 | 30 |
| C. Capacidad + Dropout | 48 h | 20.50 | 25.96 | 3.54 % | 0.894 | 26 |
| *Persistencia (línea base)* | – | 35.30 | 44.23 | 5.99 % | 0.692 | – |

MAE y RMSE en MW. Modelo recomendado (mejor en validación): **B. LSTM profunda, ventana 24 h**. El análisis completo y las respuestas a las 9 preguntas están al final de `02_lstm_modeling.ipynb`.
