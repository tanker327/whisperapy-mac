from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "whisperapy-mac"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Model
    model_repo: str = "mlx-community/whisper-large-v3-turbo"
    default_language: str = "auto"

    # Streaming (realtime WebSocket transcription)
    # Audio is expected as raw 16-bit signed PCM, mono, at this sample rate.
    stream_sample_rate: int = 16000
    # Run a transcription pass after this much new audio accumulates.
    stream_window_seconds: float = 5.0
    # Force-commit the buffer once it grows past this (Whisper's native window).
    stream_max_buffer_seconds: float = 30.0

    # File Handling
    max_file_size_mb: int = 1500
    temp_dir: Path = Path("/tmp/whisperapy")
    allowed_extensions: list[str] = [
        "mp4",
        "mov",
        "mkv",
        "avi",
        "webm",
        "mp3",
        "wav",
        "m4a",
        "ogg",
        "flac",
        "aac",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
