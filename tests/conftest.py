"""Shared fixtures.

``make_client`` builds a FastAPI app with every MLX-backed service mocked and a
fresh ``JobGate`` per app (asyncio primitives must not leak across event
loops). Tests never touch mlx-whisper, mlx-embeddings, ffmpeg, or the network.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import app.dependencies as deps
from app.config import Settings
from app.schemas.transcription import Segment, TranscribeResponse
from app.services.embedder import EmbedResult
from app.services.media import DecodedAudio

MP3_BYTES = b"ID3" + b"\x00" * 200


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings with temp directory pointing to pytest tmp."""
    return Settings(
        app_name="test-whisperapy",
        debug=True,
        temp_dir=tmp_path / "tmp",
        max_file_size_mb=10,
        _env_file=None,
    )


def sample_response(**overrides) -> TranscribeResponse:
    base = dict(
        job_id="test-123",
        text="Hello world",
        language_detected="en",
        duration_seconds=5.0,
        processing_time_seconds=1.0,
        segments=[Segment(start=0.0, end=5.0, text="Hello world")],
    )
    base.update(overrides)
    return TranscribeResponse(**base)


def decoded_audio(seconds: float = 5.0) -> DecodedAudio:
    return DecodedAudio(samples=np.zeros(int(seconds * 16000), dtype=np.float32))


@dataclass
class Env:
    client: AsyncClient
    settings: Settings
    transcriber: MagicMock
    media: MagicMock
    embedder: MagicMock


def build_mocks(audio_seconds: float = 5.0) -> tuple[MagicMock, MagicMock, MagicMock]:
    transcriber = MagicMock()
    transcriber.is_ready.return_value = True
    transcriber.transcribe.return_value = sample_response()

    media = MagicMock()

    def fake_extract_and_load(inp: Path, out: Path) -> DecodedAudio:
        out.write_bytes(b"fake wav")
        return decoded_audio(audio_seconds)

    media.extract_and_load.side_effect = fake_extract_and_load

    embedder = MagicMock()
    embedder.is_ready.return_value = True
    embedder.count_tokens.side_effect = lambda texts: [5 for _ in texts]
    embedder.embed.side_effect = lambda texts, prompt=None, dimensions=None: (
        EmbedResult(
            vectors=[[0.1, 0.2, 0.3] for _ in texts],
            prompt_tokens=5 * len(texts),
            truncated=0,
        )
    )
    return transcriber, media, embedder


@asynccontextmanager
async def app_env(settings: Settings, audio_seconds: float = 5.0) -> AsyncIterator[Env]:
    from app.main import create_app

    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    transcriber, media, embedder = build_mocks(audio_seconds)
    with (
        patch.object(deps, "_transcriber", transcriber),
        patch.object(deps, "_media_service", media),
        patch.object(deps, "_embedder", embedder),
        patch.object(deps, "_gate", deps.build_gate(settings)),
        # create_app() reads settings directly; endpoints resolve them through
        # ``Depends(get_settings)``, which needs a dependency override keyed by
        # the original (unpatched) function.
        patch("app.main.get_settings", return_value=settings),
    ):
        app = create_app()
        app.dependency_overrides[deps.get_settings] = lambda: settings
        # Let the app's own 500 handler answer instead of re-raising into tests.
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield Env(c, settings, transcriber, media, embedder)


@pytest.fixture
def make_client(tmp_path: Path) -> Callable[..., AsyncIterator[Env]]:
    """Factory: ``async with make_client(api_key="x") as env: ...``"""

    def _make(audio_seconds: float = 5.0, **settings_overrides):
        settings = Settings(
            debug=True,
            temp_dir=tmp_path / "tmp",
            max_file_size_mb=10,
            _env_file=None,
            **settings_overrides,
        )
        return app_env(settings, audio_seconds)

    return _make


@pytest.fixture
async def env(make_client) -> AsyncIterator[Env]:
    async with make_client() as e:
        yield e


@pytest.fixture
async def client(env: Env) -> AsyncClient:
    return env.client
