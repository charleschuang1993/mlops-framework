# Use a minimal Alpine-based image
FROM python:3.10-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    mlflow==2.12.0 \
    scikit-learn==1.3.0 \
    numpy==1.24.3 \
    fastapi==0.95.2 \
    uvicorn==0.22.0 \
    pydantic==1.10.7

# Copy only the necessary source files
COPY src/mlops_framework /app/src/mlops_framework
COPY serving/main.py /app/serving/main.py

EXPOSE 8000

# Run the application
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]