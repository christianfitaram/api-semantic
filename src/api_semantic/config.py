from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEFAULT_MAX_TEXT_LENGTH = 8192
DEFAULT_MAX_BATCH_SIZE = 64


class Settings(BaseSettings):
    api_key: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = DEFAULT_EMBEDDING_DEVICE
    models_dir: Path = Path("models")
    max_text_length: int = Field(default=DEFAULT_MAX_TEXT_LENGTH, gt=0)
    max_batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, gt=0)

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
