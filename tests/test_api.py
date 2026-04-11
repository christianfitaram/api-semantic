import numpy as np
import pytest
from fastapi.testclient import TestClient

import api_semantic.main as main
from api_semantic.config import get_settings


class _DummySettings:
    embedding_model = "test-model"


class _DummyEmbeddingService:
    settings = _DummySettings()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([[0.1, 0.2] for _ in texts], dtype=np.float32)


def _make_client(monkeypatch) -> TestClient:
    monkeypatch.setenv("API_KEY", "test-api-key")
    get_settings.cache_clear()
    monkeypatch.setattr(main, "get_embedding_service", lambda: _DummyEmbeddingService())
    return TestClient(main.app)


def test_health_returns_ok(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_rejects_missing_api_key(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.post("/v1/embed", json={"text": "hello world"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_embed_returns_vectors(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={"text": "hello world"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "test-model"
    assert len(payload["embedding"]) == 1
    assert payload["embedding"][0] == pytest.approx([0.1, 0.2])


def test_embed_rejects_empty_payload(monkeypatch) -> None:
    client = _make_client(monkeypatch)
    response = client.post(
        "/v1/embed",
        headers={"X-API-Key": "test-api-key"},
        json={},
    )

    assert response.status_code == 422
