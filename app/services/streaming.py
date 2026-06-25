import asyncio

import numpy as np

from app.config import Settings
from app.services.transcriber import TranscriberService

# How much trailing committed text to feed back as decoding context.
_PROMPT_MAX_CHARS = 200


class StreamingSession:
    """Stateful, single-connection accumulator for realtime transcription.

    mlx-whisper has no incremental decoding API, so we approximate realtime by
    buffering incoming PCM and re-transcribing a rolling window. After each pass
    the last segment is treated as still-forming (emitted as a ``partial``);
    earlier segments are stable, so we commit them and trim the buffer up to the
    partial's start. The committed text is fed back as ``initial_prompt`` to keep
    decoding context across windows.

    Audio is raw 16-bit signed little-endian PCM, mono, at
    ``settings.stream_sample_rate``.
    """

    def __init__(
        self,
        transcriber: TranscriberService,
        settings: Settings,
        language: str | None = None,
    ):
        self._transcriber = transcriber
        self._sample_rate = settings.stream_sample_rate
        self._window_samples = int(settings.stream_window_seconds * self._sample_rate)
        self._max_samples = int(settings.stream_max_buffer_seconds * self._sample_rate)
        self.set_language(language)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._leftover = b""  # odd trailing byte from a split PCM sample
        self._base_offset = 0.0  # absolute time (s) of buffer[0]
        self._received = 0  # total samples ever received (monotonic)
        self._last_processed = 0  # value of _received at last pass
        self._committed: list[dict] = []
        self._language_detected: str | None = None

    def set_language(self, language: str | None) -> None:
        self._language = language if language and language != "auto" else None

    def add_pcm(self, data: bytes) -> None:
        """Append raw PCM bytes to the buffer."""
        if not data:
            return
        data = self._leftover + data
        usable = len(data) - (len(data) % 2)
        self._leftover = data[usable:]
        if usable == 0:
            return
        pcm = np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32)
        pcm /= 32768.0
        self._buffer = np.concatenate([self._buffer, pcm])
        self._received += len(pcm)

    def should_process(self) -> bool:
        """True once a full window of new audio has arrived since the last pass."""
        return (self._received - self._last_processed) >= self._window_samples

    async def process(self, final: bool = False) -> tuple[list[dict], dict | None]:
        """Transcribe the current buffer.

        Returns ``(committed, partial)`` where ``committed`` are newly stabilized
        segments (absolute timestamps) and ``partial`` is the still-forming tail
        segment, if any. Inference runs off the event loop.
        """
        self._last_processed = self._received
        if len(self._buffer) == 0:
            return [], None

        result = await asyncio.to_thread(
            self._transcriber.transcribe_array,
            self._buffer,
            language=self._language,
            initial_prompt=self._prompt(),
        )

        if self._language_detected is None:
            self._language_detected = result.get("language")
            # Lock onto the detected language so later windows stay consistent.
            if self._language is None and self._language_detected:
                self._language = self._language_detected

        segments = result.get("segments", [])
        if not segments:
            self._trim_if_oversized(0.0)
            return [], None

        def to_absolute(seg: dict) -> dict:
            return {
                "start": round(self._base_offset + seg["start"], 3),
                "end": round(self._base_offset + seg["end"], 3),
                "text": seg["text"],
            }

        if final:
            committed = [to_absolute(s) for s in segments]
            self._committed.extend(committed)
            self._reset_buffer()
            return committed, None

        # Commit everything but the last (possibly mid-utterance) segment.
        stable, last = segments[:-1], segments[-1]
        committed = [to_absolute(s) for s in stable]
        self._committed.extend(committed)
        partial = to_absolute(last)

        if stable:
            self._trim(last["start"])
        else:
            # Single in-progress segment: cap unbounded buffer growth.
            self._trim_if_oversized(last["start"])

        return committed, partial

    def summary(self) -> dict:
        """Final aggregate message for the whole session."""
        return {
            "type": "done",
            "language_detected": self._language_detected,
            "duration_seconds": self._committed[-1]["end"] if self._committed else None,
            "text": " ".join(s["text"] for s in self._committed),
            "segments": self._committed,
        }

    def _prompt(self) -> str | None:
        if not self._committed:
            return None
        text = " ".join(s["text"] for s in self._committed)
        return text[-_PROMPT_MAX_CHARS:]

    def _trim(self, rel_seconds: float) -> None:
        """Drop the leading ``rel_seconds`` of buffer, advancing the offset."""
        samples = int(rel_seconds * self._sample_rate)
        if samples <= 0:
            return
        self._buffer = self._buffer[samples:]
        self._base_offset += samples / self._sample_rate

    def _trim_if_oversized(self, fallback_rel_seconds: float) -> None:
        """Keep the buffer bounded; trim to the fallback point when too large."""
        if len(self._buffer) <= self._max_samples:
            return
        self._trim(fallback_rel_seconds or self._max_samples / self._sample_rate)

    def _reset_buffer(self) -> None:
        self._base_offset += len(self._buffer) / self._sample_rate
        self._buffer = np.zeros(0, dtype=np.float32)
