"""Startup must load both models on the MLX worker thread, not the event loop.

MLX keeps GPU streams per thread; loading on the main thread and running
inference on a worker aborts the process. This test guards that invariant.
"""

import threading
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI

import app.dependencies as deps
from app.config import Settings
from app.main import lifespan
from app.services.embedder import EmbedderService
from app.services.transcriber import TranscriberService


async def test_models_load_on_mlx_thread(tmp_path: Path):
    settings = Settings(debug=True, temp_dir=tmp_path / "tmp", _env_file=None)
    load_threads: dict[str, str] = {}

    def fake_whisper_load(self):
        load_threads["whisper"] = threading.current_thread().name
        self._ready = True

    def fake_embed_load(self):
        load_threads["embed"] = threading.current_thread().name
        self._ready = True

    with (
        patch("app.main.get_settings", return_value=settings),
        patch.object(TranscriberService, "load", fake_whisper_load),
        patch.object(EmbedderService, "load", fake_embed_load),
        patch.object(deps, "_transcriber", None),
        patch.object(deps, "_embedder", None),
        patch.object(deps, "_gate", None),
        patch.object(deps, "_mlx_worker", None),
    ):
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
