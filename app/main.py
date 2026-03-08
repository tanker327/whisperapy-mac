import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.v1.router import router as v1_router
from app.core.error_handler import register_error_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestIDMiddleware, TimingMiddleware
from app.dependencies import get_settings, init_services

_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    global _start_time
    settings = get_settings()
    setup_logging(settings)

    # Create temp directory
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Starting {settings.app_name} v{settings.version}")
    transcriber = init_services(settings)
    transcriber.load()
    _start_time = time.time()
    logger.info("Server ready")

    yield

    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    # Middleware (order matters — outermost first)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    register_error_handlers(app)

    # Routes
    app.include_router(v1_router)

    @app.get("/health")
    async def health():
        from app.dependencies import get_transcriber

        transcriber = get_transcriber()
        return {
            "status": "ok",
            "version": settings.version,
            "model": settings.model_repo.split("/")[-1],
            "model_loaded": transcriber.is_ready(),
            "uptime_seconds": round(time.time() - _start_time),
        }

    @app.get("/health/model")
    async def health_model():
        from app.dependencies import get_transcriber

        transcriber = get_transcriber()
        return {
            "model_repo": settings.model_repo,
            "model_loaded": transcriber.is_ready(),
            "default_language": settings.default_language,
        }

    return app


app = create_app()
