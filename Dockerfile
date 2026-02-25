FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy only dependency/package metadata first for better layer caching.
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

VOLUME ["/app/models"]

CMD ["uvicorn", "api_semantic.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
