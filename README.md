# API Semantic

FastAPI + Poetry service that:

- converts text to embedding vectors

## Defaults

- `DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"`
- `DEFAULT_EMBEDDING_DEVICE = "cpu"`
- model cache directory: `models/`

## Setup

```bash
poetry install
cp .env.example .env
```

## Bootstrap model files

```bash
poetry run python -m api_semantic.bootstrap
```

This downloads/caches the embedding model inside `models/`.

## Run API

```bash
poetry run uvicorn api_semantic.main:app --host 0.0.0.0 --port 8000
```

## Authentication

All `/v1/*` endpoints require header:

```text
X-API-Key: <API_KEY from .env>
```

## Endpoints

- `POST /v1/embed` - text to vectors
- `GET /health` - health check

## Example requests

```bash
curl -X POST "http://localhost:8000/v1/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-this-api-key" \
  -d '{"text":"hello world"}'
```
# api-semantic
