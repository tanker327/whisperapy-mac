import sys
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.exceptions import EmbeddingError, ModelNotReadyError
from app.services.embedder import EmbedderService


@pytest.fixture
def mock_mlx_embeddings():
    """Mock the mlx_embeddings module."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"mlx_embeddings": mock}):
        yield mock


@pytest.fixture
def embedder_service():
    settings = Settings(_env_file=None)
    return EmbedderService(settings)


def test_not_ready_before_load(embedder_service):
    assert embedder_service.is_ready() is False


def test_embed_raises_when_not_ready(embedder_service):
    with pytest.raises(ModelNotReadyError):
        embedder_service.embed(["hello"])


def test_load_sets_ready(mock_mlx_embeddings, embedder_service):
    mock_mlx_embeddings.load.return_value = (MagicMock(), MagicMock())
    embedder_service.load()
    assert embedder_service.is_ready() is True
    mock_mlx_embeddings.load.assert_called_once_with(embedder_service._model_repo)


def test_embed_returns_vectors(mock_mlx_embeddings, embedder_service):
    embedder_service._ready = True
    embedder_service._tokenizer = MagicMock()
    embedder_service._tokenizer.encode.return_value = [1, 2, 3]

    output = MagicMock()
    output.text_embeds.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_mlx_embeddings.generate.return_value = output

    vectors, tokens = embedder_service.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert tokens == 6  # 3 tokens per text
    call_kwargs = mock_mlx_embeddings.generate.call_args[1]
    assert call_kwargs["texts"] == ["hello", "world"]


def test_embed_applies_prompt(mock_mlx_embeddings, embedder_service):
    embedder_service._ready = True
    embedder_service._tokenizer = MagicMock()
    embedder_service._tokenizer.encode.return_value = []

    output = MagicMock()
    output.text_embeds.tolist.return_value = [[0.1]]
    mock_mlx_embeddings.generate.return_value = output

    embedder_service.embed(["query"], prompt="Instruct: ")
    call_kwargs = mock_mlx_embeddings.generate.call_args[1]
    assert call_kwargs["texts"] == ["Instruct: query"]


def test_embed_wraps_failure(mock_mlx_embeddings, embedder_service):
    embedder_service._ready = True
    embedder_service._tokenizer = MagicMock()
    mock_mlx_embeddings.generate.side_effect = RuntimeError("boom")

    with pytest.raises(EmbeddingError):
        embedder_service.embed(["hello"])


def test_embed_token_count_falls_back_to_zero(mock_mlx_embeddings, embedder_service):
    """If the tokenizer can't encode, usage tokens degrade to 0, not an error."""
    embedder_service._ready = True
    embedder_service._tokenizer = MagicMock()
    embedder_service._tokenizer.encode.side_effect = RuntimeError("no encode")

    output = MagicMock()
    output.text_embeds.tolist.return_value = [[0.1]]
    mock_mlx_embeddings.generate.return_value = output

    vectors, tokens = embedder_service.embed(["hello"])
    assert vectors == [[0.1]]
    assert tokens == 0


def test_get_embedder_raises_when_not_initialized():
    """get_embedder() guards against use before init_services()."""
    import app.dependencies as deps

    with patch.object(deps, "_embedder", None):
        with pytest.raises(RuntimeError, match="EmbedderService not initialized"):
            deps.get_embedder()
