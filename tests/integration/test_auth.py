from tests.conftest import MP3_BYTES


async def test_no_key_configured_means_open(env):
    response = await env.client.get("/api/v1/models")
    assert response.status_code == 200


async def test_missing_key_is_401(make_client):
    async with make_client(api_key="secret") as env:
        response = await env.client.get("/api/v1/models")
        assert response.status_code == 401
        assert response.json()["error"] == "AuthenticationError"


async def test_wrong_key_is_401(make_client):
    async with make_client(api_key="secret") as env:
        response = await env.client.post(
            "/api/v1/embeddings",
            json={"input": "x"},
            headers={"Authorization": "Bearer nope"},
        )
        assert response.status_code == 401
        env.embedder.embed.assert_not_called()


async def test_bearer_key_accepted(make_client):
    async with make_client(api_key="secret") as env:
        response = await env.client.post(
            "/api/v1/embeddings",
            json={"input": "x"},
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200


async def test_x_api_key_header_accepted(make_client):
    async with make_client(api_key="secret") as env:
        response = await env.client.post(
            "/api/v1/transcribe",
            files={"file": ("a.mp3", MP3_BYTES, "audio/mpeg")},
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 200


async def test_openai_route_is_protected_too(make_client):
    async with make_client(api_key="secret") as env:
        response = await env.client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("a.mp3", MP3_BYTES, "audio/mpeg")},
        )
        assert response.status_code == 401
