import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_semantic.main as main
from api_semantic.config import get_settings
from api_semantic.embeddings import EmbeddingServiceError


class _DummySettings:
    embedding_model = "test-model"


class _DummyEmbeddingService:
    settings = _DummySettings()
    is_loaded = True
    dimension = 2

    def __init__(self) -> None:
        self.received_texts: list[str] = []
        self.load_called = False

    def load(self) -> None:
        self.load_called = True

    def embed(self, texts: list[str]) -> np.ndarray:
        self.received_texts = texts
        return np.array([[0.1, 0.2] for _ in texts], dtype=np.float32)


class _FailingEmbeddingService(_DummyEmbeddingService):
    def load(self) -> None:
        raise EmbeddingServiceError("model unavailable")

    def embed(self, texts: list[str]) -> np.ndarray:
        raise EmbeddingServiceError("model unavailable")


def _make_client(
    monkeypatch,
    *,
    service: _DummyEmbeddingService | None = None,
    max_text_length: int = 100,
    max_batch_size: int = 4,
) -> tuple[TestClient, _DummyEmbeddingService]:
    service = service or _DummyEmbeddingService()
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("MAX_TEXT_LENGTH", str(max_text_length))
    monkeypatch.setenv("MAX_BATCH_SIZE", str(max_batch_size))
    get_settings.cache_clear()
    monkeypatch.setattr(main, "get_embedding_service", lambda: service)
    return TestClient(main.app), service


def test_health_returns_ok(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_loads_model_without_api_key(monkeypatch) -> None:
    client, service = _make_client(monkeypatch)
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert service.load_called is True


def test_ready_returns_503_when_model_load_fails(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch, service=_FailingEmbeddingService())
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service unavailable"


def test_embed_rejects_missing_api_key(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch)
    response = client.post("/v1/embed", json={"text": "hello world"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_embed_rejects_invalid_api_key(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "wrong-key"},
        json={"text": "hello world"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_embed_returns_vectors(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"text": "hello world"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "test-model"
    assert payload["count"] == 1
    assert payload["dimensions"] == 2
    assert payload["normalized"] is True
    assert len(payload["embedding"]) == 1
    assert payload["embedding"][0] == pytest.approx([0.1, 0.2])


def test_embed_accepts_batch_and_trims_inputs(monkeypatch) -> None:
    client, service = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"text": " first ", "texts": [" second", "", "third "]},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 3
    assert service.received_texts == ["first", "second", "third"]


def test_embed_rejects_empty_payload(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={},
    )

    assert response.status_code == 422


def test_embed_rejects_batch_over_limit(monkeypatch) -> None:
    client, service = _make_client(monkeypatch, max_batch_size=2)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"texts": ["one", "two", "three"]},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Batch size exceeds MAX_BATCH_SIZE (2)"
    assert service.received_texts == []


def test_embed_rejects_text_over_limit(monkeypatch) -> None:
    client, service = _make_client(monkeypatch, max_text_length=5)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"text": "123456"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Text at position 1 exceeds MAX_TEXT_LENGTH (5)"
    assert service.received_texts == []


def test_embed_returns_503_when_embedding_service_fails(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch, service=_FailingEmbeddingService())
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"text": "hello world"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding service unavailable"


def test_current_model_returns_runtime_metadata(monkeypatch) -> None:
    client, _ = _make_client(monkeypatch, max_text_length=10, max_batch_size=2)
    response = client.get(
        "/v1/models/current",
        headers={"X-API-Key": "test-api-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "test-model"
    assert payload["device"] == "cpu"
    assert payload["loaded"] is True
    assert payload["dimensions"] == 2
    assert payload["max_text_length"] == 10
    assert payload["max_batch_size"] == 2
