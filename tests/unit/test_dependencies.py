from unittest.mock import patch

import pytest

import app.dependencies as deps
from app.config import Settings
from app.core.gate import JobGate
from app.core.mlx_worker import MlxWorker


@pytest.mark.parametrize(
    "attr,getter,name",
    [
        ("_transcriber", deps.get_transcriber, "TranscriberService"),
        ("_media_service", deps.get_media_service, "MediaService"),
        ("_embedder", deps.get_embedder, "EmbedderService"),
    ],
)
def test_getters_guard_against_use_before_init(attr, getter, name):
    with (
        patch.object(deps, attr, None),
        pytest.raises(RuntimeError, match=f"{name} not initialized"),
    ):
        getter()


def test_gate_and_worker_build_lazily():
    settings = Settings(max_queued_jobs=3, _env_file=None)
    with (
        patch.object(deps, "_gate", None),
        patch.object(deps, "_mlx_worker", None),
        patch.object(deps, "get_settings", return_value=settings),
    ):
        gate = deps.get_gate()
        assert isinstance(gate, JobGate) and gate.max_queued == 3
        assert deps.get_gate() is gate  # cached
        worker = deps.get_mlx_worker()
        assert isinstance(worker, MlxWorker)
        assert deps.get_mlx_worker() is worker
        worker.shutdown()


def test_init_services_builds_everything():
    settings = Settings(_env_file=None)
    with (
        patch.object(deps, "_transcriber", None),
        patch.object(deps, "_media_service", None),
        patch.object(deps, "_embedder", None),
        patch.object(deps, "_gate", None),
        patch.object(deps, "_mlx_worker", None),
    ):
        transcriber = deps.init_services(settings)
        assert deps.get_transcriber() is transcriber
        assert deps.get_media_service() is not None
        assert deps.get_embedder() is not None
        assert deps.get_gate().speed_factor == settings.transcribe_speed_factor
        deps.get_mlx_worker().shutdown()
