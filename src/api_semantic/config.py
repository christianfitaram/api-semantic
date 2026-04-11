from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DEVICE = "cpu"


class Settings(BaseSettings):
    api_key: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    models_dir: Path = Path("models")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def resolved_models_dir(self) -> Path:
        if self.models_dir.is_absolute():
            return self.models_dir
        return Path.cwd() / self.models_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
