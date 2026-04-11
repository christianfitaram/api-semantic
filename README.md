# api-semantic

`api-semantic` is a FastAPI service that converts text into normalized embedding vectors using `sentence-transformers`.

## Features

- `POST /v1/embed` endpoint for single or batch text inputs
- API key authentication on all `/v1/*` routes
- Optional model bootstrap command to warm local model cache
- Docker and Docker Compose support

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

## API

### Health

```bash
curl http://localhost:8000/health
```

### Embed

```bash
curl -X POST "http://localhost:8000/v1/embed" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-dev-key" \
  -d '{"text":"hello world"}'
```

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
