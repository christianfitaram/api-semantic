import logging
from threading import RLock

import numpy as np
from sentence_transformers import SentenceTransformer

from api_semantic.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    pass


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def load(self) -> None:
        self._load_model()

    def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                self.settings.resolved_models_dir.mkdir(parents=True, exist_ok=True)
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    cache_folder=str(self.settings.resolved_models_dir),
                )
            except Exception as exc:
                logger.exception(
                    "Failed to load embedding model",
                    extra={
                        "event": "embedding_model_load_failed",
                        "model": self.settings.embedding_model,
                        "device": self.settings.embedding_device,
                    },
                )
                raise EmbeddingServiceError("Failed to load embedding model") from exc
            return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        try:
            vectors = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            logger.exception(
                "Failed to generate embeddings",
                extra={
                    "event": "embedding_inference_failed",
                    "model": self.settings.embedding_model,
                    "batch_size": len(texts),
                },
            )
            raise EmbeddingServiceError("Failed to generate embeddings") from exc

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 2:
            self._dimension = int(array.shape[1])
        return array


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService(get_settings())
    return _service
