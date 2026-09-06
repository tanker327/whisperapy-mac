import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger

from app.config import Settings
from app.core.exceptions import ModelNotReadyError, TranscriptionError
from app.schemas.transcription import Segment, TranscribeResponse, Word

# One second of silence at 16 kHz; enough to load the model and compile the
# Metal kernels on the MLX worker thread before the first real request.
_WARMUP_AUDIO = np.zeros(16000, dtype=np.float32)


@dataclass(frozen=True)
class TranscribeOptions:
    """Subset of mlx-whisper decode options we expose over the API."""

    language: str | None = None
    word_timestamps: bool = False
    initial_prompt: str | None = None
    temperature: float | tuple[float, ...] = field(
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )
    condition_on_previous_text: bool = True

    def to_kwargs(self) -> dict:
        kwargs: dict = {
            "word_timestamps": self.word_timestamps,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if self.language and self.language != "auto":
            kwargs["language"] = self.language
        if self.initial_prompt:
            kwargs["initial_prompt"] = self.initial_prompt
        return kwargs


class TranscriberService:
    """Singleton wrapper around mlx-whisper."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_repo = settings.model_repo
        self._ready = False

    def load(self) -> None:
        """Load and warm up the model. Called at startup on the MLX thread.

        Runs a real (silent) transcription so the weights download, the model
        is cached inside mlx-whisper, and the GPU kernels are compiled. A
        failure here must propagate: a server that cannot load its model
        should not report itself ready.
        """
        import mlx_whisper

        logger.info(f"Loading model: {self._model_repo}")
        start = time.perf_counter()
        mlx_whisper.transcribe(_WARMUP_AUDIO, path_or_hf_repo=self._model_repo)
        self._ready = True
        logger.info(f"Model loaded in {time.perf_counter() - start:.1f}s")

    def is_ready(self) -> bool:
        return self._ready

    def transcribe(
        self,
        audio: np.ndarray,
        duration_seconds: float | None = None,
        options: TranscribeOptions | None = None,
    ) -> TranscribeResponse:
        """Transcribe decoded 16 kHz mono float32 audio."""
        if not self._ready:
            raise ModelNotReadyError()

        import mlx_whisper

        opts = options or TranscribeOptions()
        start = time.perf_counter()
        job_id = str(uuid.uuid4())

        try:
            result: dict[str, Any] = mlx_whisper.transcribe(
                audio, path_or_hf_repo=self._model_repo, **opts.to_kwargs()
            )
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        processing_time = time.perf_counter() - start
        segments = [_segment_from(s, opts.word_timestamps) for s in result["segments"]]
        text = str(result.get("text") or " ".join(s.text for s in segments)).strip()

        if duration_seconds is None:
            duration_seconds = segments[-1].end if segments else None

        return TranscribeResponse(
            job_id=job_id,
            language_detected=_opt_str(result.get("language")),
            duration_seconds=duration_seconds,
            processing_time_seconds=round(processing_time, 2),
            text=text,
            segments=segments,
        )


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _segment_from(raw: dict[str, Any], with_words: bool) -> Segment:
    words = (
        [
            Word(
                start=w["start"],
                end=w["end"],
                word=w["word"],
                probability=w.get("probability"),
            )
            for w in raw.get("words", [])
        ]
        if with_words
        else []
    )
    return Segment(
        start=raw["start"], end=raw["end"], text=raw["text"].strip(), words=words
    )
