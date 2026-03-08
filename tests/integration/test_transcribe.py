from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.schemas.transcription import Segment, TranscribeResponse


@pytest.fixture
async def client(tmp_path):
    """Create test client with mocked services."""
    import app.dependencies as deps
    from app.main import create_app

    settings = Settings(
        debug=True,
        temp_dir=tmp_path / "tmp",
        max_file_size_mb=10,
        _env_file=None,
    )
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    mock_transcriber = MagicMock()
    mock_transcriber.is_ready.return_value = True
    mock_transcriber.transcribe.return_value = TranscribeResponse(
        job_id="test-123",
        text="Hello world",
        language_detected="en",
        duration_seconds=5.0,
        processing_time_seconds=1.0,
        segments=[Segment(start=0.0, end=5.0, text="Hello world")],
    )

    mock_media = MagicMock()
    mock_media.extract_audio.side_effect = lambda inp, out: out.write_bytes(b"fake wav")

    with (
        patch.object(deps, "_transcriber", mock_transcriber),
        patch.object(deps, "_media_service", mock_media),
        patch.object(deps, "get_settings", return_value=settings),
    ):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_sync_transcribe(client):
    """POST /api/v1/transcribe with a valid file returns transcription."""
    # mp3 magic bytes: ID3
    file_content = b"ID3" + b"\x00" * 200
    response = await client.post(
        "/api/v1/transcribe",
        files={"file": ("test.mp3", file_content, "audio/mpeg")},
        data={"language": "auto"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["job_id"] == "test-123"
    assert data["language_detected"] == "en"
    assert data["segments"] == []


@pytest.mark.asyncio
async def test_sync_transcribe_with_segments(client):
    """POST /api/v1/transcribe with include_segments=true returns segments."""
    file_content = b"ID3" + b"\x00" * 200
    response = await client.post(
        "/api/v1/transcribe",
        files={"file": ("test.mp3", file_content, "audio/mpeg")},
        data={"include_segments": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hello world"


@pytest.mark.asyncio
async def test_transcribe_unsupported_format(client):
    """POST /api/v1/transcribe with unsupported format returns 415."""
    response = await client.post(
        "/api/v1/transcribe",
        files={"file": ("test.exe", b"\x00" * 100, "application/octet-stream")},
    )
    assert response.status_code == 415


