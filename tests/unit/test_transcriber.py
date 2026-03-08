import sys
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.exceptions import ModelNotReadyError
from app.services.transcriber import TranscriberService


@pytest.fixture
def mock_mlx_whisper():
    """Mock mlx_whisper module."""
    mock = MagicMock()
    with patch.dict(sys.modules, {"mlx_whisper": mock}):
        yield mock


@pytest.fixture
def transcriber_service():
    settings = Settings(_env_file=None)
    return TranscriberService(settings)


def test_not_ready_before_load(transcriber_service):
    assert transcriber_service.is_ready() is False


def test_transcribe_raises_when_not_ready(transcriber_service):
    with pytest.raises(ModelNotReadyError):
        transcriber_service.transcribe("audio.wav")


def test_transcribe_returns_response(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True

    mock_mlx_whisper.transcribe.return_value = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Hello world"},
        ],
    }

    result = transcriber_service.transcribe("test.wav")
    assert result.text == "Hello world"
    assert result.language_detected == "en"
    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world"


def test_transcribe_with_language(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {
        "text": "Bonjour",
        "language": "fr",
        "segments": [],
    }

    transcriber_service.transcribe("test.wav", language="fr")
    call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
    assert call_kwargs["language"] == "fr"
