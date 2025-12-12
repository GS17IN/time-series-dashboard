📊 Time-Series Forecasting Dashboard
ARIMA • SARIMAX • LSTM | Cross-Domain Comparative Analysis

This project presents an end-to-end time-series forecasting system that compares three major forecasting models — ARIMA, SARIMAX, and LSTM — across multiple heterogeneous real-world datasets, including:

🌡 Daily Temperature Data

🚗 Hourly Traffic Flow Data

💹 Stock Market Closing Prices

❤️ ECG RR-Interval Physiological Signals

The dashboard is deployed as an interactive web app (Dash + Plotly), allowing real-time visualization, forecasting, and comparison of all models.

Installation & Running Locally

    cd time-series-dashboard

Create a virtual environment (recommended)

    python -m venv venv
    source venv/bin/activate   # macOS/Linux
    venv\Scripts\activate      # Windows

Install dependencies

    pip install -r requirements.txt

Run the dashboard

    python app.py

The app will run at: http://127.0.0.1:8051


