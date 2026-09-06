from pathlib import Path

from app.config import Settings


def test_default_settings():
    """Settings should have sensible defaults."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "whisperapy-mac"
    assert settings.version == "1.0.0"
    assert settings.debug is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.max_file_size_mb == 1500
    assert "mp4" in settings.allowed_extensions
    assert "mp3" in settings.allowed_extensions


def test_settings_from_env(monkeypatch):
    """Settings should read from environment variables."""
    monkeypatch.setenv("APP_NAME", "custom-name")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "100")

    settings = Settings(_env_file=None)
    assert settings.app_name == "custom-name"
    assert settings.debug is True
    assert settings.port == 9000
    assert settings.max_file_size_mb == 100


def test_temp_dir_is_path():
    """temp_dir should be a Path object."""
    settings = Settings(_env_file=None)
    assert isinstance(settings.temp_dir, Path)


def test_concurrency_defaults():
    settings = Settings(_env_file=None)
    assert settings.max_queued_jobs == 1
    assert settings.queue_wait_seconds == 15.0
    assert settings.transcribe_speed_factor == 8.0
