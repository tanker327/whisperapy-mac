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
    embedding_model_repo: str = "mlx-community/Qwen3-Embedding-4B"

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
