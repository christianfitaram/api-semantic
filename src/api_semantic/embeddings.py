from threading import RLock

import numpy as np
from sentence_transformers import SentenceTransformer

from api_semantic.config import Settings, get_settings


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            self.settings.resolved_models_dir.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
                cache_folder=str(self.settings.resolved_models_dir),
            )
            return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService(get_settings())
    return _service
