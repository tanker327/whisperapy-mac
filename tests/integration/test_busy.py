"""End-to-end behaviour when the GPU is already busy."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

from tests.conftest import MP3_BYTES, sample_response


def slow_transcriber(env):
    """Make env's transcriber block until ``release`` is set."""
    release = threading.Event()
    started = threading.Event()

    def slow_transcribe(audio, duration_seconds=None, options=None):
        started.set()
        release.wait(timeout=5)
        return sample_response(job_id="slow", segments=[])

    env.transcriber.transcribe.side_effect = slow_transcribe
    return started, release


async def _post_transcribe(client):
    return await client.post(
        "/api/v1/transcribe", files={"file": ("a.mp3", MP3_BYTES, "audio/mpeg")}
    )


async def test_health_stays_responsive_and_reports_busy(make_client):
    async with make_client(queue_wait_seconds=5.0) as env:
        started, release = slow_transcriber(env)
        try:
            first = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.to_thread(started.wait, 2)

            health = await asyncio.wait_for(env.client.get("/health"), timeout=1)
            assert health.status_code == 200
            assert health.json()["busy"] is True
            assert health.json()["active_jobs"] == 1
            # 5 s of audio at 8x => a small, known estimate.
            assert health.json()["estimated_wait_seconds"] is not None

            release.set()
            assert (await first).status_code == 200
            assert (await env.client.get("/health")).json()["busy"] is False
        finally:
            release.set()


async def test_pipeline_full_fails_fast_with_503(make_client):
    async with make_client(queue_wait_seconds=5.0) as env:
        started, release = slow_transcriber(env)
        try:
            first = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.to_thread(started.wait, 2)

            second = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.sleep(0.05)

            third = await asyncio.wait_for(_post_transcribe(env.client), timeout=1)
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
        finally:
            release.set()


async def test_queue_wait_timeout_returns_503(make_client):
    async with make_client(queue_wait_seconds=0.2) as env:
        started, release = slow_transcriber(env)
        try:
            first = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.to_thread(started.wait, 2)

            second = await asyncio.wait_for(_post_transcribe(env.client), timeout=2)
            assert second.status_code == 503
            assert "Retry-After" in second.headers

            release.set()
            assert (await first).status_code == 200
        finally:
            release.set()


async def test_long_job_rejects_before_download(make_client):
    """URL requests are rejected up front when the running job is long."""
    async with make_client(audio_seconds=3600, queue_wait_seconds=15.0) as env:
        started, release = slow_transcriber(env)
        try:
            first = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.to_thread(started.wait, 2)

            download = MagicMock()
            with patch("app.api.v1.transcribe.download_file_from_url", download):
                resp = await asyncio.wait_for(
                    env.client.post(
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
        finally:
            release.set()


async def test_embeddings_share_the_gate(make_client):
    async with make_client(queue_wait_seconds=5.0) as env:
        started, release = slow_transcriber(env)
        try:
            first = asyncio.create_task(_post_transcribe(env.client))
            await asyncio.to_thread(started.wait, 2)

            embed = asyncio.create_task(
                env.client.post("/api/v1/embeddings", json={"input": "hi"})
            )
            await asyncio.sleep(0.05)
            health = await env.client.get("/health")
            assert health.json()["queued_jobs"] == 1

            release.set()
            r1, r2 = await asyncio.gather(first, embed)
            assert r1.status_code == 200
            assert r2.status_code == 200
        finally:
            release.set()


async def test_long_embed_job_rejects_transcription_up_front(make_client):
    """Embed jobs now carry an estimate, so a huge batch is visible to the gate."""
    async with make_client(queue_wait_seconds=1.0, embed_tokens_per_second=1) as env:
        release = threading.Event()
        started = threading.Event()

        def slow_embed(texts, prompt=None, dimensions=None):
            started.set()
            release.wait(timeout=5)
            return env.embedder.embed.side_effect_original(texts)

        env.embedder.embed.side_effect_original = env.embedder.embed.side_effect
        env.embedder.embed.side_effect = slow_embed
        env.embedder.count_tokens.side_effect = lambda texts: [5000 for _ in texts]
        try:
            embed = asyncio.create_task(
                env.client.post("/api/v1/embeddings", json={"input": "long"})
            )
            await asyncio.to_thread(started.wait, 2)

            resp = await asyncio.wait_for(_post_transcribe(env.client), timeout=1)
            assert resp.status_code == 503
            assert resp.json()["retry_after_seconds"] >= 1000

            release.set()
            assert (await embed).status_code == 200
        finally:
            release.set()
