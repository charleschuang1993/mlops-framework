FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 直接 pip install，完全不跑 apt-get
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        mlflow==2.12.0 \
        scikit-learn==1.3.0 \
        numpy==1.24.3 \
        fastapi==0.95.2 \
        uvicorn==0.22.0 \
        pydantic==1.10.7

COPY src/mlops_framework /app/src/mlops_framework
COPY src/cli.py /app/src/cli.py
COPY serving/main.py      /app/serving/main.py

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]