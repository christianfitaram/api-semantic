FROM python:3.11-slim AS builder

ARG POETRY_VERSION=2.2.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

# Copy only dependency/package metadata first for better layer caching.
COPY pyproject.toml poetry.lock README.md ./

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}" \
    && poetry install --only main --no-root --no-ansi

COPY src ./src

RUN poetry run pip install --no-deps .


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

VOLUME ["/app/models"]

CMD ["sh", "-c", "uvicorn api_semantic.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
