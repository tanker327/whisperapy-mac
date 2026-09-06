from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _package_version() -> str:
    """Single source of truth for the version: pyproject.toml via metadata."""
    try:
        return _pkg_version("whisperapy-mac")
    except PackageNotFoundError:  # running from a checkout that isn't installed
        return "0.0.0+local"


class Settings(BaseSettings):
    # extra="ignore" so a .env carrying retired keys (HOST, PORT, VERSION, ...)
    # does not stop the server from booting.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # App
    app_name: str = "whisperapy-mac"
    debug: bool = False

    @property
    def version(self) -> str:
        """Read-only: comes from package metadata, never from the environment."""
        return _package_version()

    # Security. When ``api_key`` is set, every /api/v1 route requires
    # ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Health stays open.
    api_key: str | None = None
    cors_origins: list[str] = []

    # Logging
    log_format: Literal["text", "json"] = "text"
    log_file: Path | None = None
    log_rotation: str = "50 MB"
    log_retention: str = "14 days"

    # Transcription model
    model_repo: str = "mlx-community/whisper-large-v3-turbo"

    # Embedding model
    embedding_model_repo: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    # Inputs longer than this are rejected with 422 (OpenAI behaviour) unless
    # ``embedding_truncate`` is true, in which case they are cut and flagged.
    embedding_max_tokens: int = 8192
    embedding_truncate: bool = False
    # Maximum number of strings in one request, and how many go to the GPU
    # per forward pass.
    embedding_max_batch: int = 2048
    embedding_batch_size: int = 32

    # Concurrency — the GPU runs exactly one model call at a time (MLX streams
    # are per-thread; see app/core/mlx_worker.py). Extra requests wait briefly
    # in a small queue, then fail fast with 503 + Retry-After.
    max_queued_jobs: int = 1
    queue_wait_seconds: float = 15.0
    # Rough transcription speed as a multiple of real time; drives Retry-After.
    transcribe_speed_factor: float = 8.0
    # Rough embedding throughput; drives Retry-After while an embed job runs.
    embed_tokens_per_second: float = 2000.0

    # File handling
    max_file_size_mb: int = 1500
    temp_dir: Path = Path("/tmp/whisperapy")
    # Files in temp_dir older than this are deleted at startup (crash leftovers).
    temp_max_age_hours: float = 6.0
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

    # ffmpeg extraction timeout: a floor plus a per-input-megabyte allowance,
    # so a 4K video does not trip the same limit as a podcast.
    ffmpeg_timeout_seconds: float = 120.0
    ffmpeg_timeout_seconds_per_mb: float = 0.5

    # URL downloads
    download_connect_timeout: float = 15.0
    download_read_timeout: float = 120.0
    # Allow fetching loopback / private / link-local addresses. Off by default
    # because the URL endpoint would otherwise act as an SSRF proxy.
    allow_private_urls: bool = False

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ffmpeg_timeout_for(self, input_bytes: int) -> float:
        mb = input_bytes / (1024 * 1024)
        return self.ffmpeg_timeout_seconds + mb * self.ffmpeg_timeout_seconds_per_mb
