import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.core.exceptions import ModelNotReadyError, TranscriptionError
from app.services.transcriber import TranscribeOptions, TranscriberService

AUDIO = np.zeros(16000, dtype=np.float32)


@pytest.fixture
def mock_mlx_whisper():
    mock = MagicMock()
    with patch.dict(sys.modules, {"mlx_whisper": mock}):
        yield mock


@pytest.fixture
def transcriber_service():
    return TranscriberService(Settings(_env_file=None))


def test_not_ready_before_load(transcriber_service):
    assert transcriber_service.is_ready() is False


def test_transcribe_raises_when_not_ready(transcriber_service):
    with pytest.raises(ModelNotReadyError):
        transcriber_service.transcribe(AUDIO)


def test_load_warms_up_with_real_silent_audio(mock_mlx_whisper, transcriber_service):
    mock_mlx_whisper.transcribe.return_value = {"segments": [], "text": ""}
    transcriber_service.load()
    assert transcriber_service.is_ready() is True
    args, kwargs = mock_mlx_whisper.transcribe.call_args
    assert isinstance(args[0], np.ndarray) and args[0].dtype == np.float32
    assert kwargs["path_or_hf_repo"] == transcriber_service._model_repo


def test_load_failure_propagates_and_stays_not_ready(
    mock_mlx_whisper, transcriber_service
):
    """A model that failed to load must never report itself ready."""
    mock_mlx_whisper.transcribe.side_effect = RuntimeError("no network")
    with pytest.raises(RuntimeError, match="no network"):
        transcriber_service.load()
    assert transcriber_service.is_ready() is False


def test_transcribe_returns_response(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {
        "text": " Hello world ",
        "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": " Hello world"}],
    }
    result = transcriber_service.transcribe(AUDIO, duration_seconds=3.0)
    assert result.text == "Hello world"
    assert result.language_detected == "en"
    assert result.duration_seconds == 3.0  # exact, from the decoded audio
    assert result.segments[0].text == "Hello world"
    assert result.segments[0].words == []
    assert result.processing_time_seconds is not None


def test_transcribe_passes_options(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {"text": "", "segments": []}
    opts = TranscribeOptions(
        language="fr",
        word_timestamps=True,
        initial_prompt="Names: Zoë",
        temperature=0.0,
        condition_on_previous_text=False,
    )
    transcriber_service.transcribe(AUDIO, options=opts)
    kwargs = mock_mlx_whisper.transcribe.call_args[1]
    assert kwargs["language"] == "fr"
    assert kwargs["word_timestamps"] is True
    assert kwargs["initial_prompt"] == "Names: Zoë"
    assert kwargs["temperature"] == 0.0
    assert kwargs["condition_on_previous_text"] is False


def test_transcribe_auto_language_is_not_forwarded(
    mock_mlx_whisper, transcriber_service
):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {"text": "", "segments": []}
    transcriber_service.transcribe(AUDIO, options=TranscribeOptions(language="auto"))
    assert "language" not in mock_mlx_whisper.transcribe.call_args[1]


def test_transcribe_word_timestamps(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {
        "text": "Hi there",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hi there",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "Hi", "probability": 0.9},
                    {"start": 0.5, "end": 1.0, "word": "there", "probability": 0.8},
                ],
            }
        ],
    }
    result = transcriber_service.transcribe(
        AUDIO, options=TranscribeOptions(word_timestamps=True)
    )
    assert [w.word for w in result.segments[0].words] == ["Hi", "there"]
    assert result.segments[0].words[0].probability == 0.9


def test_transcribe_duration_falls_back_to_last_segment(
    mock_mlx_whisper, transcriber_service
):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.return_value = {
        "text": "a",
        "segments": [{"start": 0.0, "end": 7.5, "text": "a"}],
    }
    assert transcriber_service.transcribe(AUDIO).duration_seconds == 7.5


def test_transcribe_wraps_failures(mock_mlx_whisper, transcriber_service):
    transcriber_service._ready = True
    mock_mlx_whisper.transcribe.side_effect = RuntimeError("metal error")
    with pytest.raises(TranscriptionError, match="metal error"):
        transcriber_service.transcribe(AUDIO)
