import logging
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request, status

from api_semantic.auth import require_api_key
from api_semantic.config import Settings, get_settings
from api_semantic.embeddings import EmbeddingServiceError, get_embedding_service
from api_semantic.logging import configure_logging
from api_semantic.schemas import (
    CurrentModelResponse,
    EmbedRequest,
    EmbedResponse,
)

configure_logging()

logger = logging.getLogger(__name__)
app = FastAPI(title="API Semantic", version="0.1.0")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = perf_counter()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "Request completed",
            extra={
                "event": "http_request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code if response else 500,
                "duration_ms": duration_ms,
                "client_host": request.client.host if request.client else None,
            },
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/v1/models/current",
    response_model=CurrentModelResponse,
    dependencies=[Depends(require_api_key)],
)
def current_model(settings: Settings = Depends(get_settings)) -> CurrentModelResponse:
    service = get_embedding_service()
    return CurrentModelResponse(
        model=settings.embedding_model,
        device=settings.embedding_device,
        loaded=service.is_loaded,
        dimensions=service.dimension,
        cache_dir=str(settings.resolved_models_dir),
        max_text_length=settings.max_text_length,
        max_batch_size=settings.max_batch_size,
    )


@app.post(
    "/v1/embed",
    response_model=EmbedResponse,
    dependencies=[Depends(require_api_key)],
)
def embed(req: EmbedRequest, settings: Settings = Depends(get_settings)) -> EmbedResponse:
    texts = _normalize_texts(req)

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text input provided",
        )

    _enforce_request_limits(texts, settings)

    service = get_embedding_service()
    try:
        vectors = service.embed(texts)
    except EmbeddingServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable",
        ) from exc

    if vectors.ndim != 2:
        logger.error(
            "Embedding service returned an invalid vector shape",
            extra={
                "event": "embedding_invalid_shape",
                "model": service.settings.embedding_model,
                "shape": list(vectors.shape),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service returned an invalid response",
        )

    dimensions = int(vectors.shape[1])
    logger.info(
        "Embeddings generated",
        extra={
            "event": "embeddings_generated",
            "model": service.settings.embedding_model,
            "count": len(texts),
            "dimensions": dimensions,
        },
    )

    return EmbedResponse(
        model=service.settings.embedding_model,
        count=len(texts),
        dimensions=dimensions,
        normalized=True,
        embedding=vectors.tolist(),
    )


def _normalize_texts(req: EmbedRequest) -> list[str]:
    texts: list[str] = []
    if req.text and req.text.strip():
        texts.append(req.text.strip())
    if req.texts:
        texts.extend([text.strip() for text in req.texts if text and text.strip()])
    return texts


def _enforce_request_limits(texts: list[str], settings: Settings) -> None:
    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Batch size exceeds MAX_BATCH_SIZE ({settings.max_batch_size})",
        )

    oversized_position = next(
        (index + 1 for index, text in enumerate(texts) if len(text) > settings.max_text_length),
        None,
    )
    if oversized_position is not None:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Text at position {oversized_position} exceeds "
                f"MAX_TEXT_LENGTH ({settings.max_text_length})"
            ),
        )
