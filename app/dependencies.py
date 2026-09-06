from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings
from app.core.gate import JobGate
from app.core.mlx_worker import MlxWorker
from app.services.embedder import EmbedderService
from app.services.media import MediaService
from app.services.transcriber import TranscriberService

_transcriber: TranscriberService | None = None
_media_service: MediaService | None = None
_embedder: EmbedderService | None = None
_gate: JobGate | None = None
_mlx_worker: MlxWorker | None = None


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


def get_embedder() -> EmbedderService:
    if _embedder is None:
        raise RuntimeError("EmbedderService not initialized")
    return _embedder


def get_gate() -> JobGate:
    global _gate
    if _gate is None:
        _gate = build_gate(get_settings())
    return _gate


def get_mlx_worker() -> MlxWorker:
    global _mlx_worker
    if _mlx_worker is None:
        _mlx_worker = MlxWorker()
    return _mlx_worker


def build_gate(settings: Settings) -> JobGate:
    # MLX streams are per-thread and all model calls run on the single
    # MlxWorker thread, so exactly one job can be on the GPU at a time.
    return JobGate(
        max_concurrent=1,
        max_queued=settings.max_queued_jobs,
        queue_wait_seconds=settings.queue_wait_seconds,
        speed_factor=settings.transcribe_speed_factor,
        embed_tokens_per_second=settings.embed_tokens_per_second,
    )


def init_services(settings: Settings) -> TranscriberService:
    """Initialize services at startup. Returns transcriber for lifespan."""
    global _transcriber, _media_service, _embedder, _gate, _mlx_worker
    _transcriber = TranscriberService(settings)
    _media_service = MediaService(settings)
    _embedder = EmbedderService(settings)
    _gate = build_gate(settings)
    _mlx_worker = MlxWorker()
    return _transcriber


# Annotated aliases so endpoints declare dependencies without calling
# ``Depends`` in argument defaults (ruff B008).
SettingsDep = Annotated[Settings, Depends(get_settings)]
TranscriberDep = Annotated[TranscriberService, Depends(get_transcriber)]
MediaDep = Annotated[MediaService, Depends(get_media_service)]
EmbedderDep = Annotated[EmbedderService, Depends(get_embedder)]
GateDep = Annotated[JobGate, Depends(get_gate)]
WorkerDep = Annotated[MlxWorker, Depends(get_mlx_worker)]
