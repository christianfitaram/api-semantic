from fastapi import Depends, FastAPI, HTTPException, status

from api_semantic.auth import require_api_key
from api_semantic.bootstrap import bootstrap_model
from api_semantic.embeddings import get_embedding_service
from api_semantic.schemas import (
    EmbedRequest,
    EmbedResponse,
)

app = FastAPI(title="API Semantic", version="0.1.0")


@app.on_event("startup")
def startup_event() -> None:
    bootstrap_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/embed",
    response_model=EmbedResponse,
    dependencies=[Depends(require_api_key)],
)
def embed(req: EmbedRequest) -> EmbedResponse:
    texts: list[str] = []
    if req.text and req.text.strip():
        texts.append(req.text.strip())
    if req.texts:
        texts.extend([t.strip() for t in req.texts if t and t.strip()])

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid text input provided",
        )

    vectors = get_embedding_service().embed(texts)
    return EmbedResponse(
        model=get_embedding_service().settings.embedding_model,
        embedding=vectors.tolist(),
    )
