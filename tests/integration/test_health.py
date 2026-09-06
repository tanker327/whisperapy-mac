from unittest.mock import patch

import app.dependencies as deps


async def test_health_ready(env):
    response = await env.client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == env.settings.version
    assert data["model"] == "whisper-large-v3-turbo"
    assert data["model_loaded"] is True
    assert data["embedding_model_loaded"] is True
    assert data["busy"] is False
    assert data["active_jobs"] == 0
    assert data["queued_jobs"] == 0
    assert data["estimated_wait_seconds"] is None


async def test_health_reports_starting_until_models_load(env):
    env.transcriber.is_ready.return_value = False
    response = await env.client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "starting"
    assert data["model_loaded"] is False
    assert data["embedding_model_loaded"] is True


async def test_health_before_init_services_is_503_not_500(env):
    with (
        patch.object(deps, "_transcriber", None),
        patch.object(deps, "_embedder", None),
    ):
        response = await env.client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "starting"


async def test_health_model_endpoint(env):
    response = await env.client.get("/health/model")
    assert response.status_code == 200
    data = response.json()
    assert data["model_repo"] == env.settings.model_repo
    assert data["model_loaded"] is True
    assert data["embedding_model_repo"] == env.settings.embedding_model_repo
    assert data["embedding_model_loaded"] is True
    assert data["embedding_max_tokens"] == env.settings.embedding_max_tokens


async def test_health_is_open_when_api_key_is_set(make_client):
    async with make_client(api_key="secret") as env:
        assert (await env.client.get("/health")).status_code == 200
        assert (await env.client.get("/health/model")).status_code == 200
