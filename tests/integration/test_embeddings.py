import base64
import struct

import pytest

from app.services.embedder import EmbedResult


async def test_embeddings_list_input(env):
    response = await env.client.post(
        "/api/v1/embeddings", json={"input": ["hello world", "goodbye"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert [d["index"] for d in data["data"]] == [0, 1]
    assert data["data"][0]["object"] == "embedding"
    assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert data["model"] == env.settings.embedding_model_repo
    assert data["usage"] == {"prompt_tokens": 10, "total_tokens": 10}
    assert data["truncated"] == 0


async def test_embeddings_string_input(env):
    response = await env.client.post("/api/v1/embeddings", json={"input": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    assert data["usage"]["prompt_tokens"] == 5


async def test_embeddings_base64_encoding(env):
    response = await env.client.post(
        "/api/v1/embeddings", json={"input": "hello", "encoding_format": "base64"}
    )
    assert response.status_code == 200
    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    floats = struct.unpack("<3f", base64.b64decode(encoded))
    assert [round(f, 5) for f in floats] == [0.1, 0.2, 0.3]


async def test_embeddings_unknown_encoding_is_422(env):
    response = await env.client.post(
        "/api/v1/embeddings", json={"input": "hello", "encoding_format": "hex"}
    )
    assert response.status_code == 422


async def test_embeddings_forwards_prompt_and_dimensions(env):
    await env.client.post(
        "/api/v1/embeddings",
        json={"input": "q", "prompt": "Instruct: ", "dimensions": 64},
    )
    kwargs = env.embedder.embed.call_args[1]
    assert kwargs["prompt"] == "Instruct: "
    assert kwargs["dimensions"] == 64


@pytest.mark.parametrize("dims", [1, 10000])
async def test_embeddings_dimensions_out_of_range_is_422(env, dims):
    response = await env.client.post(
        "/api/v1/embeddings", json={"input": "q", "dimensions": dims}
    )
    assert response.status_code == 422


async def test_embeddings_batch_limit(make_client):
    async with make_client(embedding_max_batch=2) as env:
        response = await env.client.post(
            "/api/v1/embeddings", json={"input": ["a", "b", "c"]}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "InvalidRequestError"
        env.embedder.embed.assert_not_called()


async def test_embeddings_over_length_is_422(env):
    from app.core.exceptions import InputTooLongError

    env.embedder.embed.side_effect = InputTooLongError("Input 0 is 9000 tokens")
    response = await env.client.post("/api/v1/embeddings", json={"input": "x" * 10})
    assert response.status_code == 422
    assert response.json()["error"] == "InputTooLongError"


async def test_embeddings_reports_truncation(env):
    env.embedder.embed.side_effect = lambda texts, prompt=None, dimensions=None: (
        EmbedResult(vectors=[[0.0]], prompt_tokens=8192, truncated=1)
    )
    response = await env.client.post("/api/v1/embeddings", json={"input": "x"})
    assert response.json()["truncated"] == 1


async def test_embeddings_estimate_uses_token_count(env):
    """Token counting happens before the GPU slot so the gate can estimate."""
    await env.client.post("/api/v1/embeddings", json={"input": ["a", "b"]})
    env.embedder.count_tokens.assert_called_once_with(["a", "b"])


@pytest.mark.parametrize(
    "payload", [{"input": []}, {"input": ""}, {"input": ["ok", "  "]}, {}]
)
async def test_embeddings_invalid_input_returns_422(env, payload):
    response = await env.client.post("/api/v1/embeddings", json=payload)
    assert response.status_code == 422


async def test_models_endpoint(env):
    response = await env.client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    ids = {m["id"] for m in data["data"]}
    assert env.settings.model_repo in ids
    assert env.settings.embedding_model_repo in ids
