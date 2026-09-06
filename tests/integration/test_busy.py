"""End-to-end behaviour when the GPU is already busy."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.schemas.transcription import TranscribeResponse

MP3 = b"ID3" + b"\x00" * 200
# 1 hour of 16kHz mono 16-bit audio, as bytes.
ONE_HOUR_WAV = 44 + 3600 * 32000


@pytest.fixture
def make_env(tmp_path):
    """Factory for an app whose transcriber blocks until ``release`` is set."""
    import app.dependencies as deps
    from app.main import create_app

    def _make(queue_wait_seconds: float, wav_bytes: int = 44):
        settings = Settings(
            debug=True,
            temp_dir=tmp_path / "tmp",
            max_file_size_mb=10,
            queue_wait_seconds=queue_wait_seconds,
            _env_file=None,
        )
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        release = threading.Event()
        started = threading.Event()

        def slow_transcribe(path, language=None):
            started.set()
            release.wait(timeout=5)
            return TranscribeResponse(
                job_id="slow",
                text="done",
                language_detected="en",
                duration_seconds=1.0,
                processing_time_seconds=1.0,
                segments=[],
            )

        mock_transcriber = MagicMock()
        mock_transcriber.is_ready.return_value = True
        mock_transcriber.transcribe.side_effect = slow_transcribe

        mock_media = MagicMock()

        def fake_extract(inp, out):
            # Sparse file: right size for the duration estimate, no real IO.
            with open(out, "wb") as f:
                f.truncate(wav_bytes)

        mock_media.extract_audio.side_effect = fake_extract

        mock_embedder = MagicMock()
        mock_embedder.is_ready.return_value = True
        mock_embedder.embed.side_effect = lambda texts, prompt=None: ([[0.0]], 1)

        patches = (
            patch.object(deps, "_transcriber", mock_transcriber),
            patch.object(deps, "_media_service", mock_media),
            patch.object(deps, "_embedder", mock_embedder),
            patch.object(deps, "_gate", deps.build_gate(settings)),
            patch.object(deps, "get_settings", return_value=settings),
        )
        for p in patches:
            p.start()
        app = create_app()
        client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        return client, started, release, patches

    return _make


async def _post_transcribe(client):
    return await client.post(
        "/api/v1/transcribe", files={"file": ("a.mp3", MP3, "audio/mpeg")}
    )


async def _run(make_env, queue_wait_seconds, body, wav_bytes=44):
    client, started, release, patches = make_env(queue_wait_seconds, wav_bytes)
    try:
        async with client:
            await body(client, started, release)
    finally:
        release.set()
        for p in patches:
            p.stop()


async def test_health_stays_responsive_and_reports_busy(make_env):
    async def body(client, started, release):
        first = asyncio.create_task(_post_transcribe(client))
        await asyncio.to_thread(started.wait, 2)

        health = await asyncio.wait_for(client.get("/health"), timeout=1)
        assert health.status_code == 200
        assert health.json()["busy"] is True
        assert health.json()["active_jobs"] == 1

        release.set()
        assert (await first).status_code == 200
        assert (await client.get("/health")).json()["busy"] is False

    await _run(make_env, 5.0, body)


async def test_pipeline_full_fails_fast_with_503(make_env):
    async def body(client, started, release):
        first = asyncio.create_task(_post_transcribe(client))
        await asyncio.to_thread(started.wait, 2)

        # Second request occupies the single queue slot.
        second = asyncio.create_task(_post_transcribe(client))
        await asyncio.sleep(0.05)

        third = await asyncio.wait_for(_post_transcribe(client), timeout=1)
        assert third.status_code == 503
        payload = third.json()
        assert payload["error"] == "ServiceBusyError"
        assert "retry" in payload["message"].lower()
        assert payload["retry_after_seconds"] >= 5
        assert third.headers["Retry-After"] == str(payload["retry_after_seconds"])

        release.set()
        r1, r2 = await asyncio.gather(first, second)
        assert r1.status_code == 200
        assert r2.status_code == 200  # queued request ran once the slot freed

    await _run(make_env, 5.0, body)


async def test_queue_wait_timeout_returns_503(make_env):
    async def body(client, started, release):
        first = asyncio.create_task(_post_transcribe(client))
        await asyncio.to_thread(started.wait, 2)

        second = await asyncio.wait_for(_post_transcribe(client), timeout=2)
        assert second.status_code == 503
        assert "Retry-After" in second.headers

        release.set()
        assert (await first).status_code == 200

    # Tiny wav => ~2s estimate, which exceeds a 0.2s budget; but the estimate
    # only applies once the job is running under run(), so the second request
    # waits, times out, and returns 503.
    await _run(make_env, 0.2, body)


async def test_long_job_rejects_before_download(make_env):
    """URL requests are rejected up front when the running job is long."""

    async def body(client, started, release):
        first = asyncio.create_task(_post_transcribe(client))
        await asyncio.to_thread(started.wait, 2)

        download = MagicMock()
        with patch("app.api.v1.transcribe.download_file_from_url", download):
            resp = await asyncio.wait_for(
                client.post(
                    "/api/v1/transcribe/url",
                    json={"url": "http://example.com/big.mp4"},
                ),
                timeout=1,
            )
        assert resp.status_code == 503
        # 1h audio at 8x => ~450s; Retry-After should reflect that.
        assert resp.json()["retry_after_seconds"] >= 400
        download.assert_not_called()

        release.set()
        assert (await first).status_code == 200

    await _run(make_env, 15.0, body, wav_bytes=ONE_HOUR_WAV)


async def test_embeddings_share_the_gate(make_env):
    async def body(client, started, release):
        first = asyncio.create_task(_post_transcribe(client))
        await asyncio.to_thread(started.wait, 2)

        embed = asyncio.create_task(
            client.post("/api/v1/embeddings", json={"input": "hi"})
        )
        await asyncio.sleep(0.05)
        health = await client.get("/health")
        assert health.json()["queued_jobs"] == 1

        release.set()
        r1, r2 = await asyncio.gather(first, embed)
        assert r1.status_code == 200
        assert r2.status_code == 200

    await _run(make_env, 5.0, body)
