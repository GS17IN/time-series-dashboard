"""
================================================================================
 TIME SERIES DASHBOARD 
 Supports: Temperature | Traffic | ECG RR Intervals | Stock Sectors
 Models: ARIMA | SARIMAX | LSTM
 Dash App with interactive dataset selection, forecast, and EDA mode.
================================================================================
"""

# ======================================================
# 1. IMPORTS
# ======================================================

import os
import io
import base64
import numpy as np
import pandas as pd
import wfdb

# ML + Stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Dash
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output
from dash import no_update

from scipy.fft import fft


# ======================================================
# 2. GENERIC HELPERS & METRICS
# ======================================================

def create_sequences_array(data_array, seq_len):
    X, y = [], []
    for i in range(len(data_array) - seq_len):
        X.append(data_array[i:i + seq_len])
        y.append(data_array[i + seq_len])
    return np.array(X), np.array(y)


def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def run_adf_test(series, title="Series"):
    series = series.dropna()
    result = adfuller(series)
    return {
        "Dataset": title,
        "ADF Statistic": round(result[0], 4),
        "p-value": round(result[1], 4),
        "Stationarity": "Stationary" if result[1] < 0.05 else "Non-Stationary — Differencing Recommended"
    }


def calc_error_metrics(y_true, y_pred, insample):
    """
    Computes RMSE, MAE, MASE, SMAPE, MAAPE.
    MAAPE will be used only for ECG display.
    """
    eps = 1e-8
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    insample = np.asarray(insample, dtype=float).flatten()

    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))

    if len(insample) > 1:
        naive_error = insample[1:] - insample[:-1]
        mae_naive = float(np.mean(np.abs(naive_error))) + eps
    else:
        mae_naive = mae + eps
    mase = mae / mae_naive

    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    smape = float(100.0 * np.mean(2.0 * np.abs(error) / denom))

    ratio = np.abs(error) / (np.abs(y_true) + eps)
    maape = float(np.mean(np.arctan(ratio)))

    return rmse, mae, mase, smape, maape


def model_insight(model_name, dataset_name, metric_dict):
    """
    Human explanation for model behavior, not ranking-based.
    (Used only when comparing all models.)
    """
    rmse = metric_dict.get("rmse")
    mae = metric_dict.get("mae")
    mase = metric_dict.get("mase")
    smape = metric_dict.get("smape")
    maape = metric_dict.get("maape")

    if dataset_name == "ECG RR Intervals":
        extra_metric = f"MAAPE: {maape:.4f}" if maape is not None else "MAAPE: -"
    else:
        extra_metric = f"SMAPE: {smape:.2f}%" if smape is not None else "SMAPE: -"

    base = ""
    if model_name == "ARIMA":
        base = (
            "ARIMA models short- to medium-range linear structure after differencing. "
            "It is good as a simple baseline when seasonality is weak or already removed."
        )
    elif model_name == "SARIMAX":
        base = (
            "SARIMAX explicitly models recurring seasonal patterns (like weekly or daily cycles) "
            "and can capture smoother seasonal behavior that you often see visually in the plots."
        )
    elif model_name == "LSTM":
        base = (
            "LSTM is a non-linear sequence model that can adapt to complex patterns and regime shifts, "
            "so it may track sudden changes better even if its long-horizon averages look smoother."
        )

    # Dataset context
    ctx = ""
    if dataset_name == "Daily Minimum Temperature":
        ctx = " Temperature has clear yearly and weekly structure, so both SARIMAX and LSTM can look strong for different horizons."
    elif dataset_name == "Traffic Volume":
        ctx = " Traffic has strong daily seasonality and rush-hour peaks — SARIMAX captures regular peaks, LSTM adapts to irregular spikes."
    elif dataset_name == "ECG RR Intervals":
        ctx = " ECG RR intervals are naturally noisy and irregular; visually a model may look rough but still have stable MAAPE on average."
    elif dataset_name == "Stock Market Sectors":
        ctx = " Stock sector indices are noisy with weak seasonality. Lower errors usually come from models that avoid overreacting to noise."

    return html.Div(
        style={
            "marginTop": "15px",
            "padding": "12px",
            "borderRadius": "10px",
            "backgroundColor": "#EAF4FF",
            "textAlign": "left",
        },
        children=[
            html.H4(f"{model_name} – Behavior on {dataset_name}", style={"color": "#004080"}),
            html.P(base + ctx, style={"fontSize": "14px", "margin": 0}),
        ],
    )


# ======================================================
# 3. DATA LOADING
# ======================================================

def load_temp_data(csv_path="daily-minimum-temperatures-in-me.csv"):
    temp_df = pd.read_csv(csv_path, on_bad_lines="skip")
    temp_df.columns = ["Date", "Temp"]
    temp_df["Date"] = pd.to_datetime(temp_df["Date"])
    temp_df.set_index("Date", inplace=True)

    temp_df["Temp"] = temp_df["Temp"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    temp_df["Temp"] = pd.to_numeric(temp_df["Temp"], errors="coerce")
    temp_df = temp_df.dropna()
    temp_df = temp_df.asfreq("D")
    temp_df["Temp"] = temp_df["Temp"].ffill()
    return temp_df


def load_traffic_data(csv_path="traffic.csv"):
    traffic_df = pd.read_csv(csv_path, parse_dates=["DateTime"])
    traffic_df = traffic_df.sort_values("DateTime")
    df_pivot = traffic_df.pivot(index="DateTime", columns="Junction", values="Vehicles")
    df_pivot = df_pivot.asfreq("h")
    df_pivot = df_pivot.ffill()
    df_pivot = df_pivot.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df_pivot.columns = df_pivot.columns.astype(str)
    return df_pivot


def load_ecg_rr(ecg_dir="ECG"):
    records = sorted([f.replace(".hea", "") for f in os.listdir(ecg_dir) if f.endswith(".hea")])
    if not records:
        raise FileNotFoundError("No .hea ECG header files found in ECG folder.")

    sample_record = records[0]
    rec_path = os.path.join(ecg_dir, sample_record)

    record = wfdb.rdrecord(rec_path)
    annotation = wfdb.rdann(rec_path, "atr")
    fs = record.fs

    ann_df = pd.DataFrame({
        "sample": annotation.sample,
        "time_sec": annotation.sample / fs,
        "symbol": annotation.symbol
    })

    valid_beats = ["N", "S", "V", "F"]
    ann_clean = ann_df[ann_df["symbol"].isin(valid_beats)].reset_index(drop=True)
    ann_clean["RR_interval"] = ann_clean["time_sec"].diff()
    rr = ann_clean["RR_interval"].dropna()

    rr_clean = rr[(rr > 0.3) & (rr < 2.0)].reset_index(drop=True)
    rr_df = pd.DataFrame({"RR": rr_clean})
    rr_df.index.name = "Beat"

    return rr_df


def load_stock_sector_timeseries(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    ts = df.groupby("Date")["Close"].mean().to_frame(name="Close")
    ts.index.name = "Date"
    return ts


# ======================================================
# 4. EDA FUNCTIONS
# ======================================================

def eda_temperature(tab):
    if tab == "Trend Plot":
        fig = px.line(temp_df, y="Temp", title="Daily Temperature Trend")
        return dcc.Graph(figure=fig)

    elif tab == "ACF & PACF":
        acf_vals = acf(temp_df["Temp"], nlags=40)
        pacf_vals = pacf(temp_df["Temp"], nlags=40)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
        fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals), row=1, col=1)
        fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals), row=1, col=2)
        fig.update_layout(title="Autocorrelation Plots")
        return dcc.Graph(figure=fig)

    elif tab == "Seasonal Decomposition":
        result = seasonal_decompose(temp_df["Temp"], model="additive", period=365)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])

        fig.add_trace(go.Scatter(y=result.observed), row=1, col=1)
        fig.add_trace(go.Scatter(y=result.trend), row=2, col=1)
        fig.add_trace(go.Scatter(y=result.seasonal), row=3, col=1)
        fig.add_trace(go.Scatter(y=result.resid), row=4, col=1)
        fig.update_layout(height=900, title="Seasonal Decomposition (Additive)")
        return dcc.Graph(figure=fig)


def eda_traffic(tab):
    df_long = traffic_df.reset_index().melt(
        id_vars="DateTime", var_name="Junction", value_name="Vehicles"
    )

    if tab == "Boxplot by Junction":
        fig = px.box(df_long, x="Junction", y="Vehicles",
                     title="Traffic Volume Distribution by Junction")
        return dcc.Graph(figure=fig)

    elif tab == "Hourly Trend":
        hourly = df_long.copy()
        hourly["Hour"] = hourly["DateTime"].dt.hour
        hourly_avg = hourly.groupby(["Hour", "Junction"])["Vehicles"].mean().reset_index()

        fig = px.line(hourly_avg, x="Hour", y="Vehicles", color="Junction",
                      markers=True, title="Average Hourly Traffic Volume per Junction")
        return dcc.Graph(figure=fig)

    elif tab == "Correlation Heatmap":
        fig = px.imshow(traffic_df.corr(), text_auto=True,
                        title="Correlation Between Junctions")
        return dcc.Graph(figure=fig)

    elif tab == "Volume":
        fig = px.line(df_long, x="DateTime", y="Vehicles", color="Junction",
                      title="Traffic Volume Across All Junctions")
        return dcc.Graph(figure=fig)


def eda_ecg(tab):
    if tab == "RR Trend":
        fig = px.line(ecg_df, y="RR", title="RR Interval Time Series")
        return dcc.Graph(figure=fig)

    elif tab == "RR Distribution":
        samples = 10  # seconds
        records = sorted([f.replace(".hea", "") for f in os.listdir("ECG") if f.endswith(".hea")])
        record = wfdb.rdrecord(os.path.join("ECG", records[0]))
        annotation = wfdb.rdann(os.path.join("ECG", record.record_name), "atr")

        df = pd.DataFrame({
            "time": np.arange(len(record.p_signal[:, 0])) / record.fs,
            "ecg": record.p_signal[:, 0]
        })

        ann_df = pd.DataFrame({
            "sample": annotation.sample,
            "time_sec": annotation.sample / record.fs,
            "symbol": annotation.symbol
        })

        mask = ann_df["time_sec"] < samples
        r_times = ann_df["time_sec"][mask]
        r_peaks = df["ecg"].iloc[ann_df["sample"][mask]].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["time"][:int(samples * record.fs)],
            y=df["ecg"][:int(samples * record.fs)],
            mode="lines",
            name="ECG Signal"
        ))
        fig.add_trace(go.Scatter(
            x=r_times, y=r_peaks, mode="markers",
            marker=dict(color="red", size=8),
            name="R-peaks"
        ))
        fig.update_layout(
            title="ECG Signal with R-peaks",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude"
        )
        return dcc.Graph(figure=fig)

    elif tab == "ACF & PACF":
        acf_vals = acf(ecg_df["RR"], nlags=50)
        pacf_vals = pacf(ecg_df["RR"], nlags=50)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))
        fig.add_trace(go.Bar(y=acf_vals), row=1, col=1)
        fig.add_trace(go.Bar(y=pacf_vals), row=1, col=2)
        fig.update_layout(title="ECG RR ACF/PACF")
        return dcc.Graph(figure=fig)


def eda_stock(tab):
    tech_df = pd.read_csv("technology_sector.csv")
    cons_df = pd.read_csv("consumer_discretionary_sector.csv")

    tech_df["Date"] = pd.to_datetime(tech_df["Date"])
    cons_df["Date"] = pd.to_datetime(cons_df["Date"])

    datasets = {
        "Technology": tech_df,
        "Consumer Discretionary": cons_df
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Technology Sector", "Consumer Sector"),
        horizontal_spacing=0.12
    )

    for idx, (sector, df) in enumerate(datasets.items(), start=1):
        df = df.sort_values("Date")
        df["Daily_Return"] = df.groupby("Name")["Close"].pct_change()
        df["Volatility"] = df.groupby("Name")["Daily_Return"].rolling(30).std().reset_index(0, drop=True)

        if tab == "Price Comparison":
            for name in df["Name"].unique():
                sub = df[df["Name"] == name]
                fig.add_trace(
                    go.Scatter(x=sub["Date"], y=sub["Close"], mode="lines", name=f"{sector}-{name}"),
                    row=1, col=idx
                )

        elif tab == "Returns":
            for name in df["Name"].unique():
                sub = df[df["Name"] == name]
                fig.add_trace(
                    go.Scatter(x=sub["Date"], y=sub["Daily_Return"], mode="lines", name=f"{sector}-{name}"),
                    row=1, col=idx
                )

        elif tab == "Distribution of Daily Returns":
            fig.add_trace(
                go.Histogram(
                    x=df["Daily_Return"].dropna(),
                    nbinsx=50,
                    name=sector,
                    opacity=0.75
                ),
                row=1, col=idx
            )

        elif tab == "Volatility":
            for name in df["Name"].unique():
                sub = df[df["Name"] == name]
                fig.add_trace(
                    go.Scatter(
                        x=sub["Date"], y=sub["Volatility"], mode="lines", name=f"{sector}-{name}"
                    ),
                    row=1, col=idx
                )

        elif tab == "Correlation Heatmap":
            pivot = df.pivot_table(values="Close", index="Date", columns="Name")
            corr = pivot.corr()
            heatmap = go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                showscale=True
            )
            fig.add_trace(heatmap, row=1, col=idx)

    fig.update_layout(
        height=650,
        title=f"{tab} — Technology vs Consumer Discretionary Sector",
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.6)"
        )
    )
    return dcc.Graph(figure=fig)


# ======================================================
# 5. TRAINING & FORECAST FUNCTIONS
# ======================================================

def forecast_univariate_lstm(series, model, scaler, seq_len, steps, freq):
    data = series.values.reshape(-1, 1)
    scaled = scaler.transform(data)
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    future_scaled = []

    for _ in range(steps):
        pred = model.predict(last_seq, verbose=0)[0][0]
        future_scaled.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)

    future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
    future_index = pd.date_range(series.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series(future, index=future_index)


# -------- Temperature models --------

def train_temp_models(temp_df):
    series = temp_df["Temp"]

    temp_arima = ARIMA(series, order=(3, 1, 2)).fit()
    temp_sarimax = SARIMAX(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)).fit(disp=False)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    SEQ_LEN = 30
    X, y = create_sequences_array(scaled, SEQ_LEN)

    lstm_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        LSTM(32),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    return {
        "arima": temp_arima,
        "sarimax": temp_sarimax,
        "lstm": lstm_model,
        "lstm_scaler": scaler,
        "lstm_seq_len": SEQ_LEN
    }


# -------- Traffic models --------

def train_traffic_arima_models(df):
    return {col: ARIMA(df[col].dropna(), order=(1, 1, 1)).fit() for col in df.columns}


def train_traffic_sarimax_models(df):
    return {
        col: SARIMAX(df[col].dropna(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)).fit(disp=False)
        for col in df.columns
    }


def train_traffic_lstm(df):
    df_feat = pd.DataFrame(index=df.index)
    df_feat["Hour"] = df_feat.index.hour

    full = pd.concat([df, df_feat], axis=1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full)

    seq_len = 24
    X, y = [], []

    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])
        y.append(scaled[i + seq_len][:len(df.columns)])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=(seq_len, full.shape[1])),
        Dense(64, activation="relu"),
        Dense(len(df.columns))
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X[:int(0.8 * len(X))], y[:int(0.8 * len(y))],
              epochs=15, batch_size=64, verbose=0)

    return {
        "model": model,
        "scaler": scaler,
        "seq_len": seq_len,
        "columns": full.columns.tolist(),
        "traffic_cols": df.columns.tolist()
    }


def forecast_traffic_arima(models_dict, df, steps):
    forecasts = {}
    for col, model in models_dict.items():
        fc = model.forecast(steps=steps)
        forecasts[col] = fc
    future_index = pd.date_range(df.index[-1], periods=steps + 1, freq="h")[1:]
    return pd.DataFrame(forecasts, index=future_index)


def forecast_traffic_sarimax(models_dict, df, steps):
    forecasts = {}
    for col, model in models_dict.items():
        fc = model.forecast(steps=steps)
        forecasts[col] = fc
    future_index = pd.date_range(df.index[-1], periods=steps + 1, freq="h")[1:]
    return pd.DataFrame(forecasts, index=future_index)


def forecast_traffic_lstm(df, bundle, steps):
    model = bundle["model"]
    scaler = bundle["scaler"]
    seq_len = bundle["seq_len"]
    cols = bundle["columns"]
    traffic_cols = bundle["traffic_cols"]

    df_feat = pd.DataFrame(index=df.index)
    df_feat["Hour"] = df_feat.index.hour
    full_df = pd.concat([df, df_feat], axis=1)

    scaled = scaler.transform(full_df)
    last_seq = scaled[-seq_len:].reshape(1, seq_len, len(cols))

    preds = []
    future_hours = []

    for i in range(steps):
        future_hr = (df.index[-1] + pd.Timedelta(hours=i + 1)).hour / 23.0
        future_hours.append(future_hr)

    for i in range(steps):
        pred = model.predict(last_seq, verbose=0)[0]
        row = np.zeros(len(cols))
        row[:len(traffic_cols)] = pred[:len(traffic_cols)]
        row[-1] = future_hours[i]
        preds.append(pred[:len(traffic_cols)])
        last_seq = np.append(last_seq[:, 1:, :], row.reshape(1, 1, -1), axis=1)

    inv_full = scaler.inverse_transform(
        np.column_stack([np.array(preds), np.array(future_hours)])
    )

    fc = pd.DataFrame(
        inv_full[:, :len(traffic_cols)],
        index=pd.date_range(df.index[-1], periods=steps + 1, freq="h")[1:],
        columns=traffic_cols
    )
    return fc


# -------- ECG models --------

def train_ecg_models(ecg_df):
    rr = ecg_df["RR"]

    arima = ARIMA(rr, order=(1, 0, 1)).fit()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(rr.values.reshape(-1, 1))

    seq_len = 32
    X, y = create_sequences_array(scaled, seq_len)

    lstm_model = Sequential([
        LSTM(64, activation="tanh", return_sequences=False, input_shape=(seq_len, 1)),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X, y, epochs=15, batch_size=64, verbose=0)

    rr_ds = rr[::10].reset_index(drop=True)
    sarimax = SARIMAX(rr_ds, order=(1, 1, 1), seasonal_order=(1, 0, 1, 50)).fit(disp=False)

    return {
        "arima": arima,
        "sarimax": sarimax,
        "lstm": lstm_model,
        "lstm_scaler": scaler,
        "lstm_seq_len": seq_len,
        "rr_downsampled": rr_ds
    }


def forecast_ecg_lstm(rr_series, model, scaler, seq_len, steps):
    values = scaler.transform(rr_series.values.reshape(-1, 1))
    last = values[-seq_len:].reshape(1, seq_len, 1)

    out = []
    for _ in range(steps):
        pred = model.predict(last, verbose=0)[0][0]
        out.append(pred)
        last = np.append(last[:, 1:, :], [[[pred]]], axis=1)

    future = scaler.inverse_transform(np.array(out).reshape(-1, 1)).flatten()
    return pd.Series(future, index=range(len(rr_series), len(rr_series) + steps))


# -------- Stock models --------

def train_stock_models(ts_df):
    series = ts_df["Close"]

    arima = ARIMA(series, order=(3, 1, 2)).fit()
    sarimax = SARIMAX(series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit(disp=False)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    seq = 30
    X, y = create_sequences_array(scaled, seq)

    lstm = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq, 1)),
        LSTM(32),
        Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X, y, epochs=20, batch_size=32, verbose=0)

    return {
        "arima": arima,
        "sarimax": sarimax,
        "lstm": lstm,
        "lstm_scaler": scaler,
        "lstm_seq_len": seq
    }


# ======================================================
# 6. LOAD DATA + TRAIN ALL MODELS ONCE
# ======================================================

print("Loading datasets & training models once...")

temp_df = load_temp_data()
traffic_df = load_traffic_data()
ecg_df = load_ecg_rr()

tech_ts = load_stock_sector_timeseries("technology_sector.csv")
cons_ts = load_stock_sector_timeseries("consumer_discretionary_sector.csv")

MODELS = {
    "Daily Minimum Temperature": train_temp_models(temp_df),
    "Traffic Volume": {
        "arima": train_traffic_arima_models(traffic_df),
        "sarimax": train_traffic_sarimax_models(traffic_df),
        "lstm": train_traffic_lstm(traffic_df),
    },
    "ECG RR Intervals": train_ecg_models(ecg_df),
    "Stock Market Sectors": {
        "Technology": train_stock_models(tech_ts),
        "Consumer Discretionary": train_stock_models(cons_ts),
    },
}

traffic_arima_models = MODELS["Traffic Volume"]["arima"]
traffic_sarimax_models = MODELS["Traffic Volume"]["sarimax"]
traffic_lstm_bundle = MODELS["Traffic Volume"]["lstm"]

DATASETS = {
    "Daily Minimum Temperature": temp_df,
    "Traffic Volume": traffic_df,
    "ECG RR Intervals": ecg_df,
    "Stock Market Sectors": "stocks",
}

FORECAST_CONFIG = {
    "Daily Minimum Temperature": {"min": 30, "max": 365, "default": 150, "step": 15},
    "Traffic Volume": {"min": 6, "max": 168, "default": 48, "step": 6},
    "ECG RR Intervals": {"min": 50, "max": 500, "default": 200, "step": 50},
    "Stock Market Sectors": {"min": 20, "max": 180, "default": 60, "step": 20},
}

MODEL_OPTIONS = ["ARIMA", "SARIMAX", "LSTM"]


# ======================================================
# 7. DASH APP LAYOUT
# ======================================================

app = Dash(__name__)
app.title = "Time Series Dashboard"

app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, Roboto, sans-serif",
        "backgroundColor": "#f7f9fc",
        "padding": "25px",
    },
    children=[
        html.H1(
            "📈 Time Series Forecast Dashboard",
            style={
                "textAlign": "center",
                "color": "#003366",
                "fontWeight": "700",
                "marginBottom": "30px",
                "fontSize": "36px",
            },
        ),

        # Dataset selection
        html.Div(
            style={
                "background": "white",
                "padding": "20px",
                "borderRadius": "12px",
                "boxShadow": "0px 3px 12px rgba(0,0,0,0.1)",
                "marginBottom": "25px",
                "textAlign": "center",
            },
            children=[
                html.Label(
                    "Select Dataset:",
                    style={"fontWeight": "600", "fontSize": "18px"},
                ),
                dcc.Dropdown(
                    id="dataset-dropdown",
                    options=[{"label": d, "value": d} for d in DATASETS.keys()],
                    value="Daily Minimum Temperature",
                    clearable=False,
                    style={"width": "60%", "margin": "auto", "marginTop": "10px"},
                ),
            ],
        ),

        # View mode
        html.Div(
            style={
                "textAlign": "center",
                "padding": "15px",
                "background": "white",
                "borderRadius": "12px",
                "boxShadow": "0px 3px 8px rgba(0,0,0,0.08)",
                "marginBottom": "25px",
            },
            children=[
                html.Label(
                    "Select View Mode:",
                    style={"fontWeight": "600", "fontSize": "18px"},
                ),
                dcc.RadioItems(
                    id="view-mode",
                    options=[
                        {"label": " 🔮 Forecast ", "value": "forecast"},
                        {"label": " 📊 EDA (Exploratory Data Analysis)", "value": "eda"},
                    ],
                    value="forecast",
                    inline=True,
                    style={"fontSize": "18px", "marginTop": "10px"},
                ),
            ],
        ),

        # ADF
        html.Div(
            id="adf-container",
            children=[
                html.Button("🔍 Run ADF Test", id="adf-btn", style={"margin": "10px", "padding": "8px"}),
                html.Div(id="adf-output")
            ]
        ),

        # EDA Tabs
        html.Div(
            id="eda-tabs-container",
            style={"marginBottom": "20px", "display": "none"},
            children=[
                dcc.Tabs(
                    id="eda-tab-selected",
                    value=None,
                    children=[],
                    style={
                        "background": "#e4ecff",
                        "borderRadius": "10px",
                        "padding": "10px",
                        "fontSize": "16px",
                    },
                )
            ],
        ),

        # Forecast controls
        html.Div(
            id="forecast-controls",
            style={
                "background": "white",
                "padding": "25px",
                "borderRadius": "12px",
                "boxShadow": "0px 3px 12px rgba(0,0,0,0.1)",
                "marginBottom": "25px",
            },
            children=[
                html.Div(
                    style={"textAlign": "center", "marginBottom": "25px"},
                    children=[
                        html.Label(
                            "Select Forecast Model:",
                            style={"fontWeight": "600", "fontSize": "18px"},
                        ),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[
                                {"label": "ARIMA", "value": "ARIMA"},
                                {"label": "SARIMAX", "value": "SARIMAX"},
                                {"label": "LSTM", "value": "LSTM"},
                                {"label": "Compare All Models", "value": "compare"},
                            ],
                            value="compare",
                            clearable=False,
                            style={
                                "width": "60%",
                                "margin": "auto",
                                "marginTop": "10px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={"width": "80%", "margin": "auto"},
                    children=[
                        html.Label(
                            "Forecast Horizon:",
                            style={"fontWeight": "600", "fontSize": "16px"},
                        ),
                        dcc.Slider(
                            id="horizon-slider",
                            min=30,
                            max=365,
                            value=150,
                            step=15,
                            marks={30: "30", 90: "90", 180: "180", 365: "365"},
                        ),
                    ],
                ),
            ],
        ),

        dcc.Loading(
            id="loading-spinner",
            type="cube",
            color="#003366",
            fullscreen=False,
            children=[
                html.Div(
                    id="comparison-output",
                    style={
                        "padding": "25px",
                        "background": "white",
                        "borderRadius": "12px",
                        "boxShadow": "0px 3px 12px rgba(0,0,0,0.08)",
                        "marginTop": "20px",
                        "width": "95%",
                        "margin": "auto",
                    },
                )
            ],
        ),

        html.Div(
            id="horizon-note",
            style={
                "textAlign": "center",
                "marginTop": "15px",
                "fontSize": "14px",
                "color": "#505050",
            },
        ),

        html.Div(
            "© 2025 Time Series Forecast Dashboard.",
            style={
                "textAlign": "center",
                "marginTop": "50px",
                "color": "#666",
                "fontSize": "14px",
            },
        ),
    ],
)


# ======================================================
# 8. CALLBACKS
# ======================================================

# Horizon slider config
@app.callback(
    Output("horizon-slider", "min"),
    Output("horizon-slider", "max"),
    Output("horizon-slider", "value"),
    Output("horizon-slider", "step"),
    Input("dataset-dropdown", "value"),
)
def update_horizon_slider(dataset):
    cfg = FORECAST_CONFIG.get(dataset) or FORECAST_CONFIG["Daily Minimum Temperature"]
    return cfg["min"], cfg["max"], cfg["default"], cfg["step"]


# Show/hide forecast controls
@app.callback(
    Output("forecast-controls", "style"),
    Input("view-mode", "value"),
)
def toggle_controls(mode):
    base = {
        "background": "white",
        "padding": "25px",
        "borderRadius": "12px",
        "boxShadow": "0px 3px 12px rgba(0,0,0,0.1)",
        "marginBottom": "25px",
    }
    if mode == "eda":
        base["display"] = "none"
    return base


# Show/hide EDA tabs
@app.callback(
    Output("eda-tabs-container", "style"),
    Input("view-mode", "value"),
)
def toggle_eda_visibility(view_mode):
    if view_mode == "eda":
        return {"marginBottom": "10px", "display": "block"}
    return {"marginBottom": "10px", "display": "none"}


# ADF container visibility
@app.callback(
    Output("adf-container", "style"),
    Input("view-mode", "value")
)
def toggle_adf_visibility(mode):
    if mode == "eda":
        return {"display": "block", "marginTop": "10px"}
    return {"display": "none"}


# ADF output logic: clear when mode/dataset changes, compute only on button click in EDA
@app.callback(
    Output("adf-output", "children"),
    Input("adf-btn", "n_clicks"),
    Input("view-mode", "value"),
    Input("dataset-dropdown", "value"),
    prevent_initial_call=True
)
def handle_adf(n, mode, dataset):
    ctx = callback_context
    if not ctx.triggered:
        return ""

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # If user switched mode or changed dataset → clear output
    if trigger_id in ["view-mode", "dataset-dropdown"]:
        return ""

    # Only compute when button is clicked AND in EDA mode
    if mode != "eda" or n is None:
        return ""

    if dataset == "Stock Market Sectors":
        rows = []
        stock_targets = {
            "Technology Sector": tech_ts["Close"],
            "Consumer Discretionary Sector": cons_ts["Close"],
        }

        for label, s in stock_targets.items():
            res = run_adf_test(s, label)
            rows.append(res)

        return html.Table(
            [html.Tr([
                html.Th("Dataset"),
                html.Th("ADF Statistic"),
                html.Th("p-value"),
                html.Th("Conclusion"),
            ])]
            + [
                html.Tr([
                    html.Td(r["Dataset"]),
                    html.Td(r["ADF Statistic"]),
                    html.Td(r["p-value"]),
                    html.Td(r["Stationarity"]),
                ])
                for r in rows
            ],
            style={
                "margin": "auto",
                "width": "70%",
                "border": "2px solid #004080",
                "padding": "12px",
                "fontSize": "18px",
                "borderRadius": "10px",
                "textAlign": "center",
                "background": "#F4F9FF",
            },
        )

    else:
        series_obj = DATASETS[dataset]
        series = series_obj.iloc[:, 0] if isinstance(series_obj, pd.DataFrame) else series_obj
        res = run_adf_test(series, dataset)

        return html.Table([
            html.Tr([
                html.Th("Dataset"), html.Th("ADF Statistic"),
                html.Th("p-value"), html.Th("Conclusion")
            ]),
            html.Tr([
                html.Td(res["Dataset"]),
                html.Td(res["ADF Statistic"]),
                html.Td(res["p-value"]),
                html.Td(res["Stationarity"]),
            ]),
        ], style={
            "margin": "auto",
            "width": "50%",
            "border": "2px solid #004080",
            "padding": "12px",
            "fontSize": "18px",
            "borderRadius": "10px",
            "textAlign": "center",
            "background": "#F4F9FF",
        })


# EDA tabs list
@app.callback(
    Output("eda-tab-selected", "children"),
    Output("eda-tab-selected", "value"),
    Input("dataset-dropdown", "value"),
    Input("view-mode", "value"),
)
def show_eda_tabs(dataset, mode):
    eda_tabs = {
        "Daily Minimum Temperature": ["Trend Plot", "ACF & PACF", "Seasonal Decomposition"],
        "Traffic Volume": ["Boxplot by Junction", "Hourly Trend", "Correlation Heatmap", "Volume"],
        "ECG RR Intervals": ["RR Trend", "RR Distribution", "ACF & PACF"],
        "Stock Market Sectors": [
            "Price Comparison",
            "Returns",
            "Distribution of Daily Returns",
            "Volatility",
            "Correlation Heatmap",
        ],
    }

    if mode != "eda":
        return no_update, no_update

    tabs = [dcc.Tab(label=t, value=t) for t in eda_tabs[dataset]]
    default_value = eda_tabs[dataset][0]
    return tabs, default_value


# Main forecast / EDA output
@app.callback(
    Output("comparison-output", "children"),
    Output("horizon-note", "children"),
    Input("dataset-dropdown", "value"),
    Input("model-dropdown", "value"),
    Input("horizon-slider", "value"),
    Input("view-mode", "value"),
    Input("eda-tab-selected", "value"),
)
def update_forecast(dataset_name, selected_model, horizon, view_mode, tab_selected):
    # EDA MODE
    if view_mode == "eda":
        if dataset_name == "Daily Minimum Temperature":
            return eda_temperature(tab_selected), "EDA Mode Active"
        elif dataset_name == "Traffic Volume":
            return eda_traffic(tab_selected), "EDA Mode Active"
        elif dataset_name == "ECG RR Intervals":
            return eda_ecg(tab_selected), "EDA Mode Active"
        elif dataset_name == "Stock Market Sectors":
            return eda_stock(tab_selected), "EDA Mode Active"

    # FORECAST MODE
    model_list = ["ARIMA", "SARIMAX", "LSTM"]
    figs = {}
    metrics = {}

    # ---------- STOCK BRANCH (Per sector) ----------
    if dataset_name == "Stock Market Sectors":
        stock_tech_models = MODELS["Stock Market Sectors"]["Technology"]
        stock_cons_models = MODELS["Stock Market Sectors"]["Consumer Discretionary"]

        sectors = {
            "Technology": (tech_ts, stock_tech_models),
            "Consumer Discretionary": (cons_ts, stock_cons_models),
        }

        freq_note = "Forecast horizon in days. Comparison across sectors."

        def stock_forecast(series, models_s, model_name):
            if model_name == "ARIMA":
                fc = models_s["arima"].forecast(steps=horizon)
            elif model_name == "SARIMAX":
                fc = models_s["sarimax"].forecast(steps=horizon)
            else:  # LSTM
                lstm_model = models_s["lstm"]
                scaler = models_s["lstm_scaler"]
                seq_len = models_s["lstm_seq_len"]
                fc = forecast_univariate_lstm(series, lstm_model, scaler, seq_len, horizon, "D")
            fc.index = pd.date_range(series.index[-1], periods=horizon + 1, freq="D")[1:]
            return fc

        # Compare layout: 2+1 stacked
        if selected_model == "compare":
            for m in model_list:
                fig = go.Figure()
                for sector_label, (ts, models_s) in sectors.items():
                    series = ts["Close"]
                    fc = stock_forecast(series, models_s, m)

                    fig.add_trace(go.Scatter(
                        x=series.index, y=series.values,
                        mode="lines", name=f"{sector_label} — Actual"
                    ))
                    fig.add_trace(go.Scatter(
                        x=fc.index, y=fc.values,
                        mode="lines", line=dict(dash="dash"),
                        name=f"{sector_label} — {m}",
                    ))
                fig.update_layout(
                    title=f"{m} Forecast – Technology & Consumer Sectors",
                    xaxis_title="Date",
                    yaxis_title="Average Close Price",
                    template="plotly_white",
                )
                figs[m] = fig

            display = html.Div([
                html.Div(dcc.Graph(figure=figs["ARIMA"]), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div(dcc.Graph(figure=figs["SARIMAX"]), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div(dcc.Graph(figure=figs["LSTM"]), style={"width": "85%", "margin": "20px auto"}),
            ])
        else:
            fig = go.Figure()
            for sector_label, (ts, models_s) in sectors.items():
                series = ts["Close"]
                fc = stock_forecast(series, models_s, selected_model)
                fig.add_trace(go.Scatter(
                    x=series.index, y=series.values,
                    mode="lines", name=f"{sector_label} — Actual"
                ))
                fig.add_trace(go.Scatter(
                    x=fc.index, y=fc.values,
                    mode="lines", line=dict(dash="dash"),
                    name=f"{sector_label} — {selected_model}",
                ))
            fig.update_layout(
                title=f"{selected_model} Forecast – Technology & Consumer Sectors",
                xaxis_title="Date",
                yaxis_title="Average Close Price",
                template="plotly_white",
            )
            display = html.Div([dcc.Graph(figure=fig)])

        # Metrics per sector & model (no ranking; strict single-model display)
        metrics_by_sector = {}
        for sector_label, (ts, models_s) in sectors.items():
            series = ts["Close"]
            test_len = min(horizon, int(len(series) * 0.2))
            train = series[:-test_len]
            test = series[-test_len:]

            metrics_by_sector[sector_label] = {}
            for m in model_list:
                if m == "ARIMA":
                    model_eval = ARIMA(train, order=(3, 1, 2)).fit()
                    fc_eval = model_eval.forecast(test_len)
                elif m == "SARIMAX":
                    model_eval = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
                    fc_eval = model_eval.forecast(test_len)
                else:  # LSTM
                    lstm_model = models_s["lstm"]
                    scaler = models_s["lstm_scaler"]
                    seq_len = models_s["lstm_seq_len"]
                    fc_eval = forecast_univariate_lstm(train, lstm_model, scaler, seq_len, test_len, "D")

                fc_eval.index = test.index
                rmse, mae, mase, smape, _ = calc_error_metrics(test.values, fc_eval.values, train.values)
                metrics_by_sector[sector_label][m] = {
                    "rmse": rmse,
                    "mae": mae,
                    "mase": mase,
                    "smape": smape,
                }

        # Build metrics table
        if selected_model == "compare":
            rows = []
            for sector_label, m_dict in metrics_by_sector.items():
                for m in model_list:
                    mm = m_dict[m]
                    rows.append(html.Tr([
                        html.Td(sector_label),
                        html.Td(m),
                        html.Td(f"{mm['rmse']:.4f}"),
                        html.Td(f"{mm['mae']:.4f}"),
                        html.Td(f"{mm['mase']:.4f}"),
                        html.Td(f"{mm['smape']:.2f}"),
                    ]))
        else:
            rows = []
            for sector_label, m_dict in metrics_by_sector.items():
                mm = m_dict[selected_model]
                rows.append(html.Tr([
                    html.Td(sector_label),
                    html.Td(selected_model),
                    html.Td(f"{mm['rmse']:.4f}"),
                    html.Td(f"{mm['mae']:.4f}"),
                    html.Td(f"{mm['mase']:.4f}"),
                    html.Td(f"{mm['smape']:.2f}"),
                ]))

        metrics_table = html.Table(
            [html.Tr([
                html.Th("Sector"),
                html.Th("Model"),
                html.Th("RMSE"),
                html.Th("MAE"),
                html.Th("MASE"),
                html.Th("SMAPE (%)"),
            ])] + rows,
            style={
                "margin": "20px auto",
                "border": "2px solid black",
                "padding": "10px",
                "fontSize": "16px",
                "width": "85%",
                "borderRadius": "10px",
                "textAlign": "center",
            },
        )

        # Compare mode: include interpretation cards
        if selected_model == "compare":
            insight_cards = html.Div([
                html.H3(
                    "Interpretation of Model Behavior on Stock Sectors",
                    style={
                        "textAlign": "center",
                        "color": "#003366",
                        "marginTop": "25px",
                        "fontWeight": "bold"
                    }
                ),

                html.Div(
                    "• ARIMA works well when the structure is mostly linear without strong seasonality. "
                    "In stock sector data, which is noisy and unpredictable, ARIMA acts as a reasonable baseline "
                    "but may struggle to capture volatility spikes or sudden trend reversals.",
                    style={
                        "padding": "12px",
                        "background": "#EAF4FF",
                        "borderRadius": "10px",
                        "marginBottom": "12px"
                    }
                ),

                html.Div(
                    "• SARIMAX can help if weekly or monthly cyclic behavior exists. "
                    "However, in stock indices, seasonality is weak or inconsistent. "
                    "This means SARIMAX may visually appear smoother but may not significantly improve forecasting accuracy.",
                    style={
                        "padding": "12px",
                        "background": "#EAF4FF",
                        "borderRadius": "10px",
                        "marginBottom": "12px"
                    }
                ),

                html.Div(
                    "• LSTM can model nonlinear behavior and short-term market fluctuations. "
                    "It may respond better to sudden jumps or volatility changes. "
                    "However, because stock markets are highly stochastic, deep learning models can also overfit without additional tuning.",
                    style={
                        "padding": "12px",
                        "background": "#EAF4FF",
                        "borderRadius": "10px",
                        "marginBottom": "12px"
                    }
                )
            ])

            return html.Div([display, dcc.Markdown("---"), metrics_table, insight_cards]), freq_note

        # Single-model mode: ONLY metrics table under graph
        return html.Div([display, dcc.Markdown("---"), metrics_table]), freq_note

    # ---------- NON-STOCK DATASETS (Temp / Traffic / ECG) ----------

    if dataset_name == "Daily Minimum Temperature":
        series = temp_df["Temp"]
        models = MODELS["Daily Minimum Temperature"]
        freq_note = "Forecast horizon in days."

        def get_fc(m):
            if m == "ARIMA":
                fc = models["arima"].forecast(horizon)
            elif m == "SARIMAX":
                fc = models["sarimax"].forecast(horizon)
            else:
                lstm = models["lstm"]
                scaler = models["lstm_scaler"]
                seq_len = models["lstm_seq_len"]
                fc = forecast_univariate_lstm(series, lstm, scaler, seq_len, horizon, "D")
            fc.index = pd.date_range(series.index[-1], periods=horizon + 1, freq="D")[1:]
            return fc

    elif dataset_name == "Traffic Volume":
        df = traffic_df
        freq_note = "Forecast horizon in hours."

        def get_fc(m):
            if m == "ARIMA":
                return forecast_traffic_arima(traffic_arima_models, df, horizon)
            elif m == "SARIMAX":
                return forecast_traffic_sarimax(traffic_sarimax_models, df, horizon)
            else:
                return forecast_traffic_lstm(df, traffic_lstm_bundle, horizon)

    elif dataset_name == "ECG RR Intervals":
        series = ecg_df["RR"]
        models = MODELS["ECG RR Intervals"]
        freq_note = "Forecast horizon in beats."

        def get_fc(m):
            if m == "ARIMA":
                fc = models["arima"].forecast(horizon)
                fc.index = range(len(series), len(series) + horizon)
                return fc
            elif m == "SARIMAX":
                rr_ds = models["rr_downsampled"]
                fc = models["sarimax"].forecast(horizon)
                fc.index = range(len(rr_ds), len(rr_ds) + horizon)
                return fc
            else:
                lstm = models["lstm"]
                scaler = models["lstm_scaler"]
                seq_len = models["lstm_seq_len"]
                return forecast_ecg_lstm(series, lstm, scaler, seq_len, horizon)
    else:
        return html.Div("Unknown dataset"), ""

    # Build forecasts & metrics
    for m in model_list:
        fc = get_fc(m)

        # Metrics
        if isinstance(fc, pd.Series):
            # Single variable
            if dataset_name == "Traffic Volume":
                # should not happen here
                metrics[m] = {"rmse": None, "mae": None, "mase": None, "smape": None, "maape": None}
            else:
                test_len = min(horizon, int(len(series) * 0.2))
                train = series[:-test_len]
                test = series[-test_len:]

                if m == "ARIMA":
                    model_eval = ARIMA(train, order=(3, 1, 2)).fit()
                    fc_eval = model_eval.forecast(test_len)
                elif m == "SARIMAX":
                    if dataset_name == "Daily Minimum Temperature":
                        model_eval = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
                    else:  # ECG downsampled SARIMAX is separate, here we just re-fit on train for comparison
                        model_eval = ARIMA(train, order=(1, 0, 1)).fit()
                    fc_eval = model_eval.forecast(test_len)
                else:  # LSTM
                    if dataset_name == "ECG RR Intervals":
                        lstm = models["lstm"]
                        scaler = models["lstm_scaler"]
                        seq_len = models["lstm_seq_len"]
                        fc_eval = forecast_ecg_lstm(train, lstm, scaler, seq_len, test_len)
                    else:
                        lstm = models["lstm"]
                        scaler = models["lstm_scaler"]
                        seq_len = models["lstm_seq_len"]
                        fc_eval = forecast_univariate_lstm(train, lstm, scaler, seq_len, test_len, "D")

                fc_eval.index = test.index
                rmse, mae, mase, smape, maape = calc_error_metrics(test.values, fc_eval.values, train.values)

                if dataset_name == "ECG RR Intervals":
                    metrics[m] = {"rmse": rmse, "mae": mae, "mase": mase, "smape": None, "maape": maape}
                else:
                    metrics[m] = {"rmse": rmse, "mae": mae, "mase": mase, "smape": smape, "maape": None}

        elif isinstance(fc, pd.DataFrame):
            # Traffic multivariate
            rmse_list, mae_list, mase_list, smape_list = [], [], [], []
            for col in fc.columns:
                test_len = min(horizon, int(len(traffic_df[col]) * 0.2))
                train = traffic_df[col][:-test_len]
                test = traffic_df[col][-test_len:]
                pred_eval = fc[col][:test_len]
                pred_eval.index = test.index
                r, a, ms, sm, _ = calc_error_metrics(test.values, pred_eval.values, train.values)
                rmse_list.append(r)
                mae_list.append(a)
                mase_list.append(ms)
                smape_list.append(sm)

            metrics[m] = {
                "rmse": float(np.mean(rmse_list)),
                "mae": float(np.mean(mae_list)),
                "mase": float(np.mean(mase_list)),
                "smape": float(np.mean(smape_list)),
                "maape": None,
            }
        else:
            metrics[m] = {"rmse": None, "mae": None, "mase": None, "smape": None, "maape": None}

        # Plots
        fig = go.Figure()

        if dataset_name == "ECG RR Intervals" and isinstance(fc, pd.Series):
            LOOKBACK = 300
            recent_actual = series[-LOOKBACK:].reset_index(drop=True)
            fig.add_trace(go.Scatter(
                x=list(range(len(recent_actual))),
                y=recent_actual,
                mode="lines",
                name="Actual (Last Window)",
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(recent_actual), len(recent_actual) + len(fc))),
                y=fc.values,
                mode="lines",
                line=dict(dash="dash", width=3),
                name=f"{m} Forecast",
            ))

        elif isinstance(fc, pd.Series):
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name="Actual",
            ))
            fig.add_trace(go.Scatter(
                x=fc.index,
                y=fc.values,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{m} Forecast",
            ))

        elif isinstance(fc, pd.DataFrame):
            for col in traffic_df.columns:
                fig.add_trace(go.Scatter(
                    x=traffic_df.index[-200:],
                    y=traffic_df[col][-200:],
                    mode="lines",
                    name=f"{col} Actual",
                ))
                fig.add_trace(go.Scatter(
                    x=fc.index,
                    y=fc[col],
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"{m} {col}",
                ))

        fig.update_layout(title=m, template="plotly_white")
        figs[m] = fig

    # Comparison layout 2+1
    if selected_model == "compare":
        display = html.Div([
            html.Div(dcc.Graph(figure=figs["ARIMA"]), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div(dcc.Graph(figure=figs["SARIMAX"]), style={"width": "49%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div(dcc.Graph(figure=figs["LSTM"]), style={"width": "85%", "margin": "20px auto"}),
        ])
    else:
        display = html.Div([dcc.Graph(figure=figs[selected_model])])

    # Metrics table (no ranking; strict single-model display)
    if dataset_name == "ECG RR Intervals":
        metric4_key = "maape"
        metric4_label = "MAAPE"
    else:
        metric4_key = "smape"
        metric4_label = "SMAPE (%)"

    if selected_model == "compare":
        header_row = html.Tr([
            html.Th("Model"),
            html.Th("RMSE"),
            html.Th("MAE"),
            html.Th("MASE"),
            html.Th(metric4_label),
        ])
        body_rows = []
        for m in model_list:
            mm = metrics.get(m, {})
            if mm.get("rmse") is None:
                continue
            if metric4_key == "smape":
                metric4_val = f"{mm[metric4_key]:.2f}" if mm[metric4_key] is not None else "-"
            else:
                metric4_val = f"{mm[metric4_key]:.4f}" if mm[metric4_key] is not None else "-"
            body_rows.append(html.Tr([
                html.Td(m, style={"fontWeight": "bold"}),
                html.Td(f"{mm['rmse']:.4f}"),
                html.Td(f"{mm['mae']:.4f}"),
                html.Td(f"{mm['mase']:.4f}"),
                html.Td(metric4_val),
            ]))
    else:
        mm = metrics.get(selected_model, {})
        header_row = html.Tr([
            html.Th("Model"),
            html.Th("RMSE"),
            html.Th("MAE"),
            html.Th("MASE"),
            html.Th(metric4_label),
        ])
        if metric4_key == "smape":
            metric4_val = f"{mm.get(metric4_key, 0):.2f}" if mm.get(metric4_key) is not None else "-"
        else:
            metric4_val = f"{mm.get(metric4_key, 0):.4f}" if mm.get(metric4_key) is not None else "-"
        body_rows = [html.Tr([
            html.Td(selected_model, style={"fontWeight": "bold"}),
            html.Td(f"{mm.get('rmse', 0):.4f}" if mm.get("rmse") is not None else "-"),
            html.Td(f"{mm.get('mae', 0):.4f}" if mm.get("mae") is not None else "-"),
            html.Td(f"{mm.get('mase', 0):.4f}" if mm.get("mase") is not None else "-"),
            html.Td(metric4_val),
        ])]

    metrics_display = html.Div([
        html.H3("Model Error Summary", style={
            "textAlign": "center",
            "marginBottom": "10px",
            "color": "#003366",
            "fontWeight": "bold",
        }),
        html.Table(
            [header_row] + body_rows,
            style={
                "margin": "20px auto",
                "border": "2px solid black",
                "padding": "10px",
                "fontSize": "16px",
                "width": "70%",
                "borderRadius": "10px",
                "textAlign": "center",
            },
        ),
    ])

    # Compare mode: include insight cards
    if selected_model == "compare":
        insight_cards = html.Div([
            html.H3("How to interpret these models", style={
                "textAlign": "center",
                "color": "#003366",
                "marginTop": "25px",
            }),
            model_insight("ARIMA", dataset_name, metrics.get("ARIMA", {})),
            model_insight("SARIMAX", dataset_name, metrics.get("SARIMAX", {})),
            model_insight("LSTM", dataset_name, metrics.get("LSTM", {})),
        ])
        return html.Div([display, dcc.Markdown("---"), metrics_display, insight_cards]), freq_note

    # Single-model mode: ONLY metrics table under the graph (no explanation text)
    return html.Div([
        display,
        dcc.Markdown("---"),
        metrics_display
    ]), freq_note


# ======================================================
# 9. RUN APP
# ======================================================

if __name__ == "__main__":
    print("Starting Dash app on http://127.0.0.1:8051")
    app.run(host="127.0.0.1", port=8051, debug=False)
