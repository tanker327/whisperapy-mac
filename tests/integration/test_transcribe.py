from unittest.mock import patch

import numpy as np
import pytest

from app.schemas.transcription import Segment
from app.services.transcriber import TranscribeOptions
from tests.conftest import MP3_BYTES, sample_response


async def post_upload(client, data=None, filename="test.mp3", content=MP3_BYTES):
    return await client.post(
        "/api/v1/transcribe",
        files={"file": (filename, content, "audio/mpeg")},
        data=data or {},
    )


async def test_upload_returns_transcript_without_segments(env):
    response = await post_upload(env.client, {"language": "auto"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["job_id"] == "test-123"
    assert data["language_detected"] == "en"
    assert data["duration_seconds"] == 5.0
    assert data["segments"] == []
    assert "X-Request-ID" in response.headers
    assert "X-Processing-Time" in response.headers


async def test_upload_with_segments(env):
    response = await post_upload(env.client, {"include_segments": "true"})
    assert response.status_code == 200
    assert response.json()["segments"][0]["text"] == "Hello world"


async def test_upload_passes_decoded_audio_and_options_to_transcriber(env):
    await post_upload(
        env.client,
        {
            "language": "de",
            "word_timestamps": "true",
            "initial_prompt": "Berlin",
            "temperature": "0.2",
            "condition_on_previous_text": "false",
        },
    )
    args, kwargs = env.transcriber.transcribe.call_args
    assert isinstance(args[0], np.ndarray)  # the decoded samples, not a path
    assert kwargs["duration_seconds"] == 5.0
    opts: TranscribeOptions = kwargs["options"]
    assert opts.language == "de"
    assert opts.word_timestamps is True
    assert opts.initial_prompt == "Berlin"
    assert opts.temperature == 0.2
    assert opts.condition_on_previous_text is False


async def test_upload_default_temperature_is_whisper_fallback_ladder(env):
    await post_upload(env.client)
    opts = env.transcriber.transcribe.call_args[1]["options"]
    assert opts.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@pytest.mark.parametrize(
    "fmt,media_type,head",
    [
        ("text", "text/plain", "Hello world"),
        (
            "srt",
            "application/x-subrip",
            "1\n00:00:00,000 --> 00:00:05,000\nHello world",
        ),
        ("vtt", "text/vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world"),
    ],
)
async def test_upload_output_formats(env, fmt, media_type, head):
    response = await post_upload(env.client, {"output_format": fmt})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media_type)
    assert response.text.startswith(head)


async def test_upload_verbose_json_always_includes_segments(env):
    response = await post_upload(env.client, {"output_format": "verbose_json"})
    assert len(response.json()["segments"]) == 1


async def test_upload_invalid_output_format_is_422(env):
    response = await post_upload(env.client, {"output_format": "docx"})
    assert response.status_code == 422


async def test_upload_unsupported_extension_is_415(env):
    response = await post_upload(env.client, filename="test.exe", content=b"\x00" * 100)
    assert response.status_code == 415
    assert response.json()["error"] == "UnsupportedFormatError"


async def test_upload_magic_mismatch_is_415(env):
    response = await post_upload(env.client, filename="fake.mp3", content=b"\x00" * 100)
    assert response.status_code == 415


async def test_upload_cleans_temp_files(env):
    await post_upload(env.client)
    assert list(env.settings.temp_dir.iterdir()) == []


async def test_upload_cleans_temp_files_when_transcription_fails(env):
    from app.core.exceptions import TranscriptionError

    env.transcriber.transcribe.side_effect = TranscriptionError("metal")
    response = await post_upload(env.client)
    assert response.status_code == 500
    assert response.json()["error"] == "TranscriptionError"
    assert list(env.settings.temp_dir.iterdir()) == []


async def test_url_endpoint(env):
    async def fake_download(url, settings):
        assert url == "http://example.com/audio.mp3"
        p = settings.temp_dir / "downloaded"
        p.write_bytes(b"fake audio")
        return p

    with patch(
        "app.api.v1.transcribe.download_file_from_url", side_effect=fake_download
    ):
        response = await env.client.post(
            "/api/v1/transcribe/url",
            json={"url": "http://example.com/audio.mp3", "include_segments": True},
        )
    assert response.status_code == 200
    assert len(response.json()["segments"]) == 1
    assert list(env.settings.temp_dir.iterdir()) == []


async def test_url_endpoint_rejects_non_http_scheme(env):
    response = await env.client.post(
        "/api/v1/transcribe/url", json={"url": "file:///etc/passwd"}
    )
    assert response.status_code == 422


async def test_url_endpoint_blocks_private_hosts(env):
    response = await env.client.post(
        "/api/v1/transcribe/url", json={"url": "http://127.0.0.1:9/x.mp3"}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "ForbiddenUrlError"


# ---------------------------------------------------------- OpenAI shape


async def post_openai(client, data=None):
    return await client.post(
        "/api/v1/audio/transcriptions",
        files={"file": ("test.mp3", MP3_BYTES, "audio/mpeg")},
        data=data or {},
    )


async def test_openai_json_default(env):
    response = await post_openai(env.client, {"model": "whisper-1"})
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["segments"] == []


async def test_openai_verbose_json_with_word_granularity(env):
    env.transcriber.transcribe.return_value = sample_response(
        segments=[Segment(start=0, end=1, text="Hi", words=[])]
    )
    response = await post_openai(
        env.client,
        {
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
            "language": "en",
            "prompt": "Hi",
        },
    )
    assert response.status_code == 200
    assert len(response.json()["segments"]) == 1
    opts = env.transcriber.transcribe.call_args[1]["options"]
    assert opts.language == "en"
    assert opts.initial_prompt == "Hi"


async def test_openai_text_format(env):
    response = await post_openai(env.client, {"response_format": "text"})
    assert response.status_code == 200
    assert response.text == "Hello world"
