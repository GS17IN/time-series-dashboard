Time Series Forecast Dashboard

A fully interactive Dash-based web application for forecasting and analyzing multiple time-series datasets.
The system supports statistical and deep learning forecasting techniques and provides EDA, stationarity checks, model comparisons, and sector-wise analysis.

The authors
    1. Namitha K Ram
    2. Geethika S

Repository Contents

├── app.py                      # Main Dash application
├── requirements.txt            # Dependencies for deployment
├── Procfile                   # Render/Heroku runtime config
├── ECG/                       # WFDB files for RR interval extraction
├── technology_sector.csv
├── consumer_discretionary_sector.csv
├── daily-minimum-temperatures-in-me.csv
├── traffic.csv
└── README.md

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


