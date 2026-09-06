from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import health
from app.api.v1.router import router as v1_router
from app.core.error_handler import register_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.dependencies import (
    get_embedder,
    get_mlx_worker,
    get_settings,
    init_services,
)
from app.utils.file_handler import sweep_temp_dir


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage startup and shutdown."""
    settings = get_settings()
    setup_logging(settings)
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    sweep_temp_dir(settings)

    # Load models on the dedicated MLX thread, never on the event loop.
    # MLX GPU streams are per-thread, so the thread that loads the models must
    # be the thread that runs inference (see app/core/mlx_worker.py). A load
    # failure propagates and aborts startup on purpose.
    transcriber = init_services(settings)
    worker = get_mlx_worker()
    await worker.run(transcriber.load)
    await worker.run(get_embedder().load)

    health.mark_started()
    logger.info("Server ready")

    yield

    logger.info("Shutting down")
    worker.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        description=(
            "Local transcription and embedding service on Apple Silicon. "
            "Native routes under /api/v1 plus OpenAI-compatible "
            "/api/v1/audio/transcriptions, /api/v1/embeddings and /api/v1/models."
        ),
    )

    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(v1_router)
    return app


app = create_app()
