from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from api_semantic.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    api_key: str | None = Depends(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
