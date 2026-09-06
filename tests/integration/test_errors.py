"""The error envelope: every domain error maps to its status and one JSON
shape; unexpected exceptions never leak a stack trace."""

import pytest

from app.core import exceptions as ex
from tests.conftest import MP3_BYTES


async def post(client):
    return await client.post(
        "/api/v1/transcribe", files={"file": ("a.mp3", MP3_BYTES, "audio/mpeg")}
    )


@pytest.mark.parametrize(
    "exc,status",
    [
        (ex.FileTooLargeError, 413),
        (ex.UnsupportedFormatError, 415),
        (ex.FileValidationError, 422),
        (ex.DownloadError, 422),
        (ex.ForbiddenUrlError, 422),
        (ex.InvalidRequestError, 422),
        (ex.InputTooLongError, 422),
        (ex.AuthenticationError, 401),
        (ex.AudioExtractionError, 500),
        (ex.TranscriptionError, 500),
        (ex.EmbeddingError, 500),
        (ex.ModelNotReadyError, 503),
    ],
)
async def test_domain_errors_map_to_status_and_envelope(env, exc, status):
    env.media.extract_and_load.side_effect = exc("boom")
    response = await post(env.client)
    assert response.status_code == status
    body = response.json()
    assert body == {
        "error": exc.__name__,
        "message": "boom",
        "request_id": response.headers["X-Request-ID"],
    }


async def test_default_messages_are_used_when_none_given(env):
    env.media.extract_and_load.side_effect = ex.AudioExtractionError()
    body = (await post(env.client)).json()
    assert body["message"] == "Audio extraction failed"


async def test_service_busy_carries_retry_after(env):
    env.media.extract_and_load.side_effect = ex.ServiceBusyError(retry_after=42)
    response = await post(env.client)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "42"
    assert response.json()["retry_after_seconds"] == 42


async def test_unhandled_exception_is_500_without_details(env):
    env.media.extract_and_load.side_effect = ZeroDivisionError("secret detail")
    response = await post(env.client)
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalServerError"
    assert "secret" not in body["message"]
    assert body["request_id"] == response.headers["X-Request-ID"]


async def test_incoming_request_id_is_honoured(env):
    response = await env.client.get("/health", headers={"X-Request-ID": "trace-1"})
    assert response.headers["X-Request-ID"] == "trace-1"


async def test_pydantic_validation_error_is_422(env):
    response = await env.client.post("/api/v1/transcribe/url", json={"url": 5})
    assert response.status_code == 422
