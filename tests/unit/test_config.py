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
    assert settings.max_file_size_mb == 500
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
