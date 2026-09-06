from pathlib import Path

from app.config import Settings


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.app_name == "whisperapy-mac"
    assert settings.debug is False
    assert settings.max_file_size_mb == 1500
    assert settings.temp_dir == Path("/tmp/whisperapy")
    assert "mp4" in settings.allowed_extensions
    assert "mp3" in settings.allowed_extensions
    assert settings.api_key is None
    assert settings.cors_origins == []
    assert settings.allow_private_urls is False


def test_version_comes_from_package_metadata():
    settings = Settings(_env_file=None)
    # Installed from pyproject via uv; a bare checkout gets the local marker.
    assert settings.version and settings.version != "1.0.0"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("APP_NAME", "custom-name")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "100")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("CORS_ORIGINS", '["http://a", "http://b"]')

    settings = Settings(_env_file=None)
    assert settings.app_name == "custom-name"
    assert settings.debug is True
    assert settings.max_file_size_mb == 100
    assert settings.api_key == "secret"
    assert settings.cors_origins == ["http://a", "http://b"]


def test_retired_env_keys_are_ignored(monkeypatch):
    """An old .env with HOST/PORT/VERSION must not stop the server booting."""
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("VERSION", "1.0.0")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "auto")
    Settings(_env_file=None)


def test_concurrency_defaults():
    settings = Settings(_env_file=None)
    assert settings.max_queued_jobs == 1
    assert settings.queue_wait_seconds == 15.0
    assert settings.transcribe_speed_factor == 8.0
    assert settings.embed_tokens_per_second == 2000.0


def test_embedding_limits_defaults():
    settings = Settings(_env_file=None)
    assert settings.embedding_max_tokens == 8192
    assert settings.embedding_truncate is False
    assert settings.embedding_max_batch == 2048
    assert settings.embedding_batch_size == 32


def test_ffmpeg_timeout_scales_with_input_size():
    settings = Settings(
        ffmpeg_timeout_seconds=100, ffmpeg_timeout_seconds_per_mb=2, _env_file=None
    )
    assert settings.ffmpeg_timeout_for(0) == 100
    assert settings.ffmpeg_timeout_for(10 * 1024 * 1024) == 120


def test_max_file_size_bytes():
    settings = Settings(max_file_size_mb=2, _env_file=None)
    assert settings.max_file_size_bytes == 2 * 1024 * 1024
