# --- Base Python Image ---
FROM python:3.10-slim

# --- System Dependencies for Pandas, Statsmodels, pmdarima, etc. ---
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Create Working Directory ---
WORKDIR /app

# --- Copy requirements first ---
COPY requirements.txt .

# --- Install Python Dependencies ---
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --- Copy the rest of your project ---
COPY . .

# --- Expose Render Port ---
EXPOSE 10000

# --- Start server ---
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:10000"]
