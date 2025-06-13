FROM python:3.10-slim

# Install poetry
RUN pip install --no-cache-dir poetry

WORKDIR /app

# Copy pyproject and install deps
COPY pyproject.toml /app/
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --only main --no-root

# Copy source
COPY serving /app/serving

EXPOSE 8000
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000"]