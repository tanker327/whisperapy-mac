from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from starlette.testclient import TestClient

from app.config import Settings


@pytest.fixture
def stream_client(tmp_path):
    """TestClient with a mocked transcriber, wired for the WebSocket endpoint."""
    from app.main import create_app

    settings = Settings(
        debug=True,
        temp_dir=tmp_path / "tmp",
        stream_sample_rate=16000,
        stream_window_seconds=1.0,
        _env_file=None,
    )

    transcriber = MagicMock()
    transcriber.is_ready.return_value = True
    transcriber.transcribe_array.return_value = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ],
    }

    # Endpoint imports these names into its own module namespace, so patch there.
    with (
        patch("app.api.v1.transcribe.get_transcriber", return_value=transcriber),
        patch("app.api.v1.transcribe.get_settings", return_value=settings),
    ):
        # No lifespan context manager → real model is never loaded.
        yield TestClient(create_app()), transcriber


def pcm_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    return np.zeros(int(seconds * sample_rate), dtype=np.int16).tobytes()


def test_stream_emits_partial_final_and_done(stream_client):
    client, _ = stream_client
    with client.websocket_connect("/api/v1/transcribe/stream") as ws:
        ws.send_bytes(pcm_bytes(2.0))  # > 1s window → triggers a pass

        first = ws.receive_json()
        assert first["type"] == "final"
        assert first["segments"][0]["text"] == "hello"

        second = ws.receive_json()
        assert second["type"] == "partial"
        assert second["segment"]["text"] == "world"

        ws.send_json({"type": "end"})

        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg["type"] == "done":
                break

        done = messages[-1]
        assert done["language_detected"] == "en"
        assert "world" in done["text"]


def test_stream_passes_configured_language_to_transcriber(stream_client):
    client, transcriber = stream_client
    with client.websocket_connect("/api/v1/transcribe/stream") as ws:
        ws.send_json({"type": "config", "language": "fr"})
        ws.send_bytes(pcm_bytes(2.0))
        ws.receive_json()  # final
        ws.receive_json()  # partial

    assert transcriber.transcribe_array.call_args.kwargs["language"] == "fr"


def test_stream_rejects_invalid_control_message(stream_client):
    client, _ = stream_client
    with client.websocket_connect("/api/v1/transcribe/stream") as ws:
        ws.send_text("not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_stream_errors_when_model_not_ready(tmp_path):
    from app.main import create_app

    settings = Settings(temp_dir=tmp_path / "tmp", _env_file=None)
    transcriber = MagicMock()
    transcriber.is_ready.return_value = False

    with (
        patch("app.api.v1.transcribe.get_transcriber", return_value=transcriber),
        patch("app.api.v1.transcribe.get_settings", return_value=settings),
    ):
        client = TestClient(create_app())
        with client.websocket_connect("/api/v1/transcribe/stream") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "not ready" in msg["message"].lower()
