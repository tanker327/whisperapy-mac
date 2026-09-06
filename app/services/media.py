"""ffmpeg wrapper: any container in, 16 kHz mono PCM out.

Whisper needs 16 kHz mono float32. We decode once with ffmpeg to a WAV on
disk (so a broken file fails before a GPU slot is taken and so we know the
exact duration), then load that WAV into a numpy array and hand the array to
mlx-whisper. Passing the *path* instead would make mlx-whisper spawn ffmpeg a
second time for the same audio.
"""

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from app.config import Settings
from app.core.exceptions import AudioExtractionError

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray  # float32 in [-1, 1]
    sample_rate: int = SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / self.sample_rate


def load_wav(path: Path) -> DecodedAudio:
    """Read a 16-bit PCM WAV produced by ``extract_audio`` into float32."""
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                raise AudioExtractionError("Extracted WAV is not 16-bit mono")
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise AudioExtractionError(f"Could not read extracted audio: {e}") from e
    pcm = np.frombuffer(frames, dtype=np.int16)
    return DecodedAudio(samples=pcm.astype(np.float32) / 32768.0, sample_rate=rate)


class MediaService:
    """Wraps ffmpeg to extract audio from any input format."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def extract_audio(self, input_path: Path, output_path: Path) -> Path:
        """Extract audio as 16kHz mono 16-bit WAV using ffmpeg."""
        try:
            input_bytes = input_path.stat().st_size
        except OSError:
            input_bytes = 0
        timeout = self._settings.ffmpeg_timeout_for(input_bytes)
        logger.info(
            f"Extracting audio: {input_path.name} -> {output_path.name} "
            f"(timeout {timeout:.0f}s)"
        )

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vn",  # no video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",  # mono
            "-y",  # overwrite
            str(output_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise AudioExtractionError(
                f"ffmpeg timed out after {timeout:.0f} seconds"
            ) from e
        except FileNotFoundError as e:
            raise AudioExtractionError("ffmpeg is not installed or not on PATH") from e
        except OSError as e:
            raise AudioExtractionError(f"Audio extraction failed: {e}") from e

        if result.returncode != 0:
            logger.error(f"ffmpeg stderr: {result.stderr}")
            raise AudioExtractionError(f"ffmpeg failed: {result.stderr[-500:]}")

        logger.info(f"Audio extracted: {output_path.name}")
        return output_path

    def extract_and_load(self, input_path: Path, output_path: Path) -> DecodedAudio:
        """Extract to WAV, then load it as a float32 array (one decode total)."""
        self.extract_audio(input_path, output_path)
        return load_wav(output_path)
