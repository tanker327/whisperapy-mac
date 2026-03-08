from functools import lru_cache

from app.config import Settings
from app.services.media import MediaService
from app.services.transcriber import TranscriberService

_transcriber: TranscriberService | None = None
_media_service: MediaService | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_transcriber() -> TranscriberService:
    if _transcriber is None:
        raise RuntimeError("TranscriberService not initialized")
    return _transcriber


def get_media_service() -> MediaService:
    if _media_service is None:
        raise RuntimeError("MediaService not initialized")
    return _media_service


def init_services(settings: Settings) -> TranscriberService:
    """Initialize services at startup. Returns transcriber for lifespan."""
    global _transcriber, _media_service
    _transcriber = TranscriberService(settings)
    _media_service = MediaService()
    return _transcriber
