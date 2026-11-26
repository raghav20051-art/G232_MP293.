FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt requirements.txt

# Install Python dependencies + FastAPI/Uvicorn
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] joblib

# Copy application source
COPY src/ src/
COPY params.yaml params.yaml

# Copy trained model artifact
COPY models/model.joblib models/model.joblib

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "src.mp293.api:app", "--host", "0.0.0.0", "--port", "8000"]
