from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings


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


@pytest.fixture
def mock_upload_file():
    """Create a mock UploadFile."""

    def _make(
        filename: str = "test.mp3",
        content: bytes = b"ID3" + b"\x00" * 100,
        size: int | None = None,
    ):
        mock = AsyncMock()
        mock.filename = filename
        mock.size = size or len(content)
        mock.read = AsyncMock(return_value=content)
        mock.seek = AsyncMock()
        return mock

    return _make


@pytest.fixture
def mock_transcriber():
    """Mock TranscriberService."""
    mock = MagicMock()
    mock.is_ready.return_value = True
    mock.transcribe.return_value = MagicMock(
        job_id="test-job-id",
        status="completed",
        language_detected="en",
        duration_seconds=10.0,
        processing_time_seconds=1.0,
        text="Hello world",
        segments=[],
    )
    return mock


@pytest.fixture
async def async_client():
    """Async test client with mocked services."""
    from unittest.mock import patch

    from app.main import create_app

    settings = Settings(
        debug=True,
        temp_dir=Path("/tmp/whisperapy-test"),
        _env_file=None,
    )

    with patch("app.dependencies.get_settings", return_value=settings):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
