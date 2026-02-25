from sentence_transformers import SentenceTransformer

from api_semantic.config import get_settings


def bootstrap_model() -> None:
    settings = get_settings()
    settings.resolved_models_dir.mkdir(parents=True, exist_ok=True)

    SentenceTransformer(
        settings.embedding_model,
        device=settings.embedding_device,
        cache_folder=str(settings.resolved_models_dir),
    )


if __name__ == "__main__":
    bootstrap_model()
    print("Model bootstrap completed.")
