# --- Base Python Image ---
FROM python:3.10-slim

# --- System Dependencies for Pandas, Statsmodels, pmdarima, etc. ---
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Create Working Directory ---
WORKDIR /app

# --- Copy requirements first for caching efficiency ---
COPY requirements.txt .

# --- Install Python Dependencies ---
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --- Copy the rest of your app ---
COPY . .

# --- Expose port for Render ---
EXPOSE 10000

# --- Run Dash App using Gunicorn ---
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:10000"]
