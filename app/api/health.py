"""Liveness and readiness.

``/health`` answers 200 once both models are loaded and 503 with
``status: "starting"`` before that, so launchd, load balancers, and clients
can tell "booting" from "broken". These routes sit outside the v1 router and
are never behind the API key.
"""

import time

from fastapi import APIRouter, Response

import app.dependencies as deps
from app.dependencies import SettingsDep

router = APIRouter(tags=["health"])

_start_time: float | None = None


def mark_started() -> None:
    global _start_time
    _start_time = time.time()


def _uptime() -> int | None:
    return None if _start_time is None else round(time.time() - _start_time)


def _models_ready() -> tuple[bool, bool]:
    transcriber_ready = deps._transcriber is not None and deps._transcriber.is_ready()
    embedder_ready = deps._embedder is not None and deps._embedder.is_ready()
    return transcriber_ready, embedder_ready


@router.get("/health")
async def health(settings: SettingsDep, response: Response) -> dict:
    transcriber_ready, embedder_ready = _models_ready()
    ready = transcriber_ready and embedder_ready
    if not ready:
        response.status_code = 503
    return {
        "status": "ok" if ready else "starting",
        "version": settings.version,
        "model": settings.model_repo.split("/")[-1],
        "model_loaded": transcriber_ready,
        "embedding_model": settings.embedding_model_repo.split("/")[-1],
        "embedding_model_loaded": embedder_ready,
        "uptime_seconds": _uptime(),
        **deps.get_gate().snapshot(),
    }


@router.get("/health/model")
async def health_model(settings: SettingsDep) -> dict:
    transcriber_ready, embedder_ready = _models_ready()
    return {
        "model_repo": settings.model_repo,
        "model_loaded": transcriber_ready,
        "embedding_model_repo": settings.embedding_model_repo,
        "embedding_model_loaded": embedder_ready,
        "embedding_max_tokens": settings.embedding_max_tokens,
    }
