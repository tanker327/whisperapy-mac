import math
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.exceptions import EmbeddingError, InputTooLongError, ModelNotReadyError
from app.services.embedder import EmbedderService, _shrink


@pytest.fixture
def mock_mlx_embeddings():
    mock = MagicMock()
    with patch.dict(sys.modules, {"mlx_embeddings": mock}):
        yield mock


def make_service(**overrides) -> EmbedderService:
    settings = Settings(_env_file=None, **overrides)
    svc = EmbedderService(settings)
    svc._ready = True
    svc._tokenizer = MagicMock()
    svc._tokenizer.encode.side_effect = lambda t: list(range(len(t.split())))
    return svc


def set_output(mock, dims=2):
    def gen(model, tokenizer, texts, **kwargs):
        out = MagicMock()
        out.text_embeds.tolist.return_value = [[0.1] * dims for _ in texts]
        return out

    mock.generate.side_effect = gen


def test_not_ready_before_load():
    assert EmbedderService(Settings(_env_file=None)).is_ready() is False


def test_embed_raises_when_not_ready():
    with pytest.raises(ModelNotReadyError):
        EmbedderService(Settings(_env_file=None)).embed(["hello"])


def test_load_sets_ready(mock_mlx_embeddings):
    svc = EmbedderService(Settings(_env_file=None))
    mock_mlx_embeddings.load.return_value = (MagicMock(), MagicMock())
    svc.load()
    assert svc.is_ready() is True
    mock_mlx_embeddings.load.assert_called_once_with(svc._model_repo)


def test_embed_returns_vectors_and_token_usage(mock_mlx_embeddings):
    svc = make_service()
    set_output(mock_mlx_embeddings)
    result = svc.embed(["hello world", "one two three"])
    assert result.vectors == [[0.1, 0.1], [0.1, 0.1]]
    assert result.prompt_tokens == 5
    assert result.truncated == 0


def test_embed_passes_max_length_from_settings(mock_mlx_embeddings):
    svc = make_service(embedding_max_tokens=1234)
    set_output(mock_mlx_embeddings)
    svc.embed(["hello"])
    kwargs = mock_mlx_embeddings.generate.call_args[1]
    assert kwargs["max_length"] == 1234
    assert kwargs["truncation"] is True


def test_embed_rejects_over_length_input_by_default(mock_mlx_embeddings):
    svc = make_service(embedding_max_tokens=2)
    set_output(mock_mlx_embeddings)
    with pytest.raises(InputTooLongError, match="Input 1 is 3 tokens"):
        svc.embed(["ok ok", "one two three"])
    mock_mlx_embeddings.generate.assert_not_called()


def test_embed_truncates_when_enabled_and_reports_it(mock_mlx_embeddings):
    svc = make_service(embedding_max_tokens=2, embedding_truncate=True)
    set_output(mock_mlx_embeddings)
    result = svc.embed(["ok ok", "one two three"])
    assert result.truncated == 1
    # Usage counts what was embedded (2 + 2), not what was sent (2 + 3).
    assert result.prompt_tokens == 4


def test_embed_processes_in_batches(mock_mlx_embeddings):
    svc = make_service(embedding_batch_size=2)
    set_output(mock_mlx_embeddings)
    result = svc.embed(["a", "b", "c", "d", "e"])
    assert mock_mlx_embeddings.generate.call_count == 3
    sizes = [len(c[1]["texts"]) for c in mock_mlx_embeddings.generate.call_args_list]
    assert sizes == [2, 2, 1]
    assert len(result.vectors) == 5


def test_embed_applies_prompt(mock_mlx_embeddings):
    svc = make_service()
    set_output(mock_mlx_embeddings)
    svc.embed(["query"], prompt="Instruct: ")
    assert mock_mlx_embeddings.generate.call_args[1]["texts"] == ["Instruct: query"]


def test_embed_dimensions_shrinks_and_renormalises(mock_mlx_embeddings):
    svc = make_service()

    def gen(model, tokenizer, texts, **kwargs):
        out = MagicMock()
        out.text_embeds.tolist.return_value = [[0.6, 0.8, 0.0, 0.0]]
        return out

    mock_mlx_embeddings.generate.side_effect = gen
    result = svc.embed(["x"], dimensions=2)
    vec = result.vectors[0]
    assert len(vec) == 2
    assert math.isclose(sum(v * v for v in vec), 1.0)


def test_shrink_handles_zero_vector():
    assert _shrink([0.0, 0.0, 0.0], 2) == [0.0, 0.0]


def test_embed_wraps_failure(mock_mlx_embeddings):
    svc = make_service()
    mock_mlx_embeddings.generate.side_effect = RuntimeError("boom")
    with pytest.raises(EmbeddingError):
        svc.embed(["hello"])


def test_token_count_falls_back_to_zero(mock_mlx_embeddings):
    svc = make_service()
    svc._tokenizer.encode.side_effect = RuntimeError("no encode")
    set_output(mock_mlx_embeddings)
    result = svc.embed(["hello"])
    assert result.prompt_tokens == 0


def test_count_tokens_without_tokenizer():
    svc = EmbedderService(Settings(_env_file=None))
    assert svc.count_tokens(["a", "b"]) == [0, 0]


def test_get_embedder_raises_when_not_initialized():
    import app.dependencies as deps

    with (
        patch.object(deps, "_embedder", None),
        pytest.raises(RuntimeError, match="EmbedderService not initialized"),
    ):
        deps.get_embedder()
