# api-semantic

`api-semantic` is a FastAPI service that converts text into normalized embedding vectors using `sentence-transformers`.

## Features

- `POST /v1/embed` endpoint for single or batch text inputs
- API key authentication on all `/v1/*` routes
- Configurable request limits for text length and batch size
- Current model metadata endpoint
- Optional model bootstrap command to warm local model cache
- Docker and Docker Compose support
- JSON structured application logs

## Design goals

This repository is the open-source core of a larger proprietary semantic system. It intentionally focuses on a small, reusable embedding API surface rather than the full private product: clear configuration, predictable request validation, controlled failure modes, and deployment-friendly packaging.

## Requirements

- Python `>=3.10,<3.13`
- Poetry `>=1.8`

## Quickstart

```bash
poetry install --with dev
cp .env.example .env
poetry run python -m api_semantic.bootstrap
poetry run uvicorn api_semantic.main:app --host 0.0.0.0 --port 8000
```

## Configuration

Configuration is loaded from environment variables (or `.env`).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `API_KEY` | Yes | - | API key required by `/v1/*` endpoints |
| `EMBEDDING_MODEL` | No | `BAAI/bge-m3` | HuggingFace model id |
| `EMBEDDING_DEVICE` | No | `cpu` | Inference device (for example `cpu`, `cuda`) |
| `MODELS_DIR` | No | `models` | Local path for model cache |
| `MAX_TEXT_LENGTH` | No | `8192` | Maximum characters allowed per text item |
| `MAX_BATCH_SIZE` | No | `64` | Maximum number of text items per request |

## API

### Health

```bash
curl http://localhost:8000/health
```

### Current model

```bash
curl "http://localhost:8000/v1/models/current" \
  -H "X-API-Key: local-dev-key"
```

Example response:

```json
{
  "model": "BAAI/bge-m3",
  "device": "cpu",
  "loaded": false,
  "dimensions": null,
  "cache_dir": "/app/models",
  "max_text_length": 8192,
  "max_batch_size": 64
}
```

### Embed

```bash
curl -X POST "http://localhost:8000/v1/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-dev-key" \
  -d '{"text":"hello world"}'
```

Single and batch inputs use the same response shape. `embedding` is always a list of vectors, even when the request contains one text item.

Example response:

```json
{
  "model": "BAAI/bge-m3",
  "count": 1,
  "dimensions": 1024,
  "normalized": true,
  "embedding": [
    [0.0123, -0.0456, 0.0789]
  ]
}
```

Large requests are rejected before model inference:

- `413` when the number of valid text items exceeds `MAX_BATCH_SIZE`
- `413` when any text item exceeds `MAX_TEXT_LENGTH`
- `503` when the configured embedding model cannot be loaded or inference fails

## Docker

Build image:

```bash
docker compose build
```

Bootstrap model files into the persistent volume:

```bash
docker compose run --rm bootstrap
```

Run API:

```bash
docker compose up api
```

The named volume is `api_semantic_models` mounted at `/app/models`.

## Development

Run lint and tests:

```bash
poetry run ruff check .
poetry run pytest
```

## Security

If you discover a vulnerability, please follow [`SECURITY.md`](SECURITY.md).

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).
