from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings


@pytest.fixture
async def client():
    """Create test client with mocked transcriber."""
    import app.dependencies as deps
    from app.main import create_app

    settings = Settings(
        debug=True,
        temp_dir=Path("/tmp/whisperapy-test"),
        _env_file=None,
    )

    mock_transcriber = MagicMock()
    mock_transcriber.is_ready.return_value = True
    mock_media = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.is_ready.return_value = True

    with (
        patch.object(deps, "_transcriber", mock_transcriber),
        patch.object(deps, "_media_service", mock_media),
        patch.object(deps, "_embedder", mock_embedder),
        patch.object(deps, "_gate", deps.build_gate(settings)),
        patch.object(deps, "get_settings", return_value=settings),
    ):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health_endpoint(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data
    assert data["embedding_model_loaded"] is True
    assert data["busy"] is False
    assert data["active_jobs"] == 0
    assert data["queued_jobs"] == 0
    assert data["estimated_wait_seconds"] is None


@pytest.mark.asyncio
async def test_health_model_endpoint(client):
    response = await client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert "model_repo" in data
    assert "model_loaded" in data
    assert "default_language" in data
    assert "embedding_model_repo" in data
    assert data["embedding_model_loaded"] is True
