import time
import uuid

from loguru import logger

from app.config import Settings
from app.core.exceptions import ModelNotReadyError, TranscriptionError
from app.schemas.transcription import Segment, TranscribeResponse


class TranscriberService:
    """Singleton wrapper around mlx-whisper."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_repo = settings.model_repo
        self._ready = False

    def load(self) -> None:
        """Load the mlx-whisper model. Called at startup."""
        import mlx_whisper

        logger.info(f"Loading model: {self._model_repo}")
        # Warm up the model by running a dummy transcription
        # mlx-whisper loads lazily on first call
        try:
            mlx_whisper.transcribe(
                str(self._settings.temp_dir / ".gitkeep"),
                path_or_hf_repo=self._model_repo,
            )
        except Exception:
            # The dummy file won't produce results, but the model gets cached
            pass
        self._ready = True
        logger.info("Model loaded successfully")

    def is_ready(self) -> bool:
        return self._ready

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscribeResponse:
        """Transcribe an audio file and return structured response."""
        if not self._ready:
            raise ModelNotReadyError()

        import mlx_whisper

        start = time.perf_counter()
        job_id = str(uuid.uuid4())

        try:
            kwargs: dict = {
                "path_or_hf_repo": self._model_repo,
            }
            if language and language != "auto":
                kwargs["language"] = language

            result = mlx_whisper.transcribe(audio_path, **kwargs)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

        processing_time = time.perf_counter() - start

        segments = [
            Segment(start=s["start"], end=s["end"], text=s["text"].strip())
            for s in result.get("segments", [])
        ]

        return TranscribeResponse(
            job_id=job_id,
            language_detected=result.get("language"),
            duration_seconds=segments[-1].end if segments else None,
            processing_time_seconds=round(processing_time, 2),
            text=" ".join(s.text for s in segments),
            segments=segments,
        )
