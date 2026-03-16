from pathlib import Path

from loguru import logger

from app.core.exceptions import AudioExtractionError


class MediaService:
    """Wraps ffmpeg to extract audio from any input format."""

    def extract_audio(self, input_path: Path, output_path: Path) -> Path:
        """Extract audio as 16kHz mono WAV using ffmpeg."""
        import subprocess

        logger.info(f"Extracting audio: {input_path.name} -> {output_path.name}")

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-vn",  # no video
                "-acodec",
                "pcm_s16le",  # 16-bit PCM
                "-ar",
                "16000",  # 16kHz sample rate
                "-ac",
                "1",  # mono
                "-y",  # overwrite
                str(output_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg stderr: {result.stderr}")
                raise AudioExtractionError(f"ffmpeg failed: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired as e:
            raise AudioExtractionError("ffmpeg timed out after 300 seconds") from e
        except AudioExtractionError:
            raise
        except Exception as e:
            raise AudioExtractionError(f"Audio extraction failed: {e}") from e

        logger.info(f"Audio extracted: {output_path.name}")
        return output_path
