# 基礎階段
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# 安裝系統依賴（測試時需要）
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴檔案
COPY requirements-test.txt .

# 安裝 Python 套件（分層以利用快取）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-test.txt

# 複製程式碼
COPY src/ /app/src/
COPY serving/ /app/serving/
COPY tests/ /app/tests/

# 測試階段
FROM base as test
# 可以在 CI 中執行 pytest

# 生產階段
FROM python:3.11-slim as production

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 只安裝生產環境套件
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# 只複製必要的程式碼
COPY src/ /app/src/
COPY serving/ /app/serving/

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]