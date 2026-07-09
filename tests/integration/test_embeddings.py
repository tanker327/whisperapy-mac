from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings


@pytest.fixture
async def client():
    """Create test client with a mocked embedder."""
    import app.dependencies as deps
    from app.main import create_app

    settings = Settings(
        debug=True,
        temp_dir=Path("/tmp/whisperapy-test"),
        _env_file=None,
    )

    mock_embedder = MagicMock()
    mock_embedder.is_ready.return_value = True
    # Return one 3-dim vector per input text, plus a token count.
    mock_embedder.embed.side_effect = lambda texts, prompt=None: (
        [[0.1, 0.2, 0.3] for _ in texts],
        len(texts) * 5,
    )

    with (
        patch.object(deps, "_embedder", mock_embedder),
        patch.object(deps, "get_settings", return_value=settings),
    ):
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_embeddings_list_input(client):
    """POST /api/v1/embeddings with a list returns OpenAI-shaped JSON."""
    response = await client.post(
        "/api/v1/embeddings",
        json={"input": ["hello world", "goodbye"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["model"] == settings_repo()
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_embeddings_string_input(client):
    """POST /api/v1/embeddings with a single string returns one vector."""
    response = await client.post(
        "/api/v1/embeddings",
        json={"input": "hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert data["data"][0]["index"] == 0
    assert data["usage"]["prompt_tokens"] == 5


def settings_repo() -> str:
    return Settings(_env_file=None).embedding_model_repo
