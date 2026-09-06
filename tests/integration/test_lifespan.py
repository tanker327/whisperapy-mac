"""Startup must load both models on the MLX worker thread, not the event loop.

MLX keeps GPU streams per thread; loading on the main thread and running
inference on a worker aborts the process. This test guards that invariant.
"""

import os
import threading
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI

import app.dependencies as deps
from app.config import Settings
from app.main import lifespan
from app.services.embedder import EmbedderService
from app.services.transcriber import TranscriberService


@contextmanager
def fresh_app_state(settings: Settings, whisper_load, embed_load) -> Iterator[None]:
    """Reset every module-level singleton and stub both model loaders."""
    with ExitStack() as stack:
        stack.enter_context(patch("app.main.get_settings", return_value=settings))
        for name in ("_transcriber", "_embedder", "_media_service", "_gate"):
            stack.enter_context(patch.object(deps, name, None))
        stack.enter_context(patch.object(deps, "_mlx_worker", None))
        stack.enter_context(patch.object(TranscriberService, "load", whisper_load))
        stack.enter_context(patch.object(EmbedderService, "load", embed_load))
        yield


def ok_load(self):
    self._ready = True


async def test_models_load_on_mlx_thread(tmp_path: Path):
    settings = Settings(debug=True, temp_dir=tmp_path / "tmp", _env_file=None)
    load_threads: dict[str, str] = {}

    def fake_whisper_load(self):
        load_threads["whisper"] = threading.current_thread().name
        self._ready = True

    def fake_embed_load(self):
        load_threads["embed"] = threading.current_thread().name
        self._ready = True

    with fresh_app_state(settings, fake_whisper_load, fake_embed_load):
        async with lifespan(FastAPI()):
            assert deps.get_transcriber().is_ready()
            assert deps.get_embedder().is_ready()
            assert settings.temp_dir.is_dir()
            # Inference goes through the same worker the models loaded on.
            infer_thread = await deps.get_mlx_worker().run(
                lambda: threading.current_thread().name
            )

    main = threading.current_thread().name
    assert load_threads["whisper"] == load_threads["embed"] == infer_thread
    assert infer_thread != main
    assert infer_thread.startswith("mlx")


async def test_model_load_failure_aborts_startup(tmp_path: Path):
    """A model that cannot load must fail the lifespan, not report healthy."""
    settings = Settings(debug=True, temp_dir=tmp_path / "tmp", _env_file=None)

    def broken_load(self):
        raise RuntimeError("weights download failed")

    with (
        fresh_app_state(settings, broken_load, ok_load),
        pytest.raises(RuntimeError, match="weights download failed"),
    ):
        async with lifespan(FastAPI()):
            pass


async def test_startup_sweeps_stale_temp_files(tmp_path: Path):
    settings = Settings(
        debug=True, temp_dir=tmp_path / "tmp", temp_max_age_hours=1, _env_file=None
    )
    settings.temp_dir.mkdir(parents=True)
    stale = settings.temp_dir / "leftover.wav"
    stale.write_bytes(b"x")
    old = time.time() - 7200
    os.utime(stale, (old, old))

    with fresh_app_state(settings, ok_load, ok_load):
        async with lifespan(FastAPI()):
            assert not stale.exists()
