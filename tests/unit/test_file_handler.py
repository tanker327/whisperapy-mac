from unittest.mock import AsyncMock

import pytest

from app.core.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFormatError,
)
from app.utils.file_handler import cleanup_temp, sanitize_filename, validate_upload


def test_sanitize_filename_strips_path():
    assert sanitize_filename("/etc/passwd") == "passwd"
    assert sanitize_filename("../../secret.txt") == "secret.txt"


def test_sanitize_filename_replaces_unsafe_chars():
    result = sanitize_filename("my file (1).mp3")
    assert " " not in result
    assert "(" not in result


def test_sanitize_filename_preserves_extension():
    result = sanitize_filename("video.mp4")
    assert result.endswith(".mp4")


@pytest.mark.asyncio
async def test_validate_upload_no_filename(test_settings):
    mock_file = AsyncMock()
    mock_file.filename = None
    with pytest.raises(FileValidationError):
        await validate_upload(mock_file, test_settings)


@pytest.mark.asyncio
async def test_validate_upload_bad_extension(test_settings):
    mock_file = AsyncMock()
    mock_file.filename = "file.exe"
    mock_file.read = AsyncMock(return_value=b"\x00" * 100)
    mock_file.seek = AsyncMock()
    with pytest.raises(UnsupportedFormatError):
        await validate_upload(mock_file, test_settings)


@pytest.mark.asyncio
async def test_validate_upload_too_large(test_settings):
    test_settings.max_file_size_mb = 1  # 1MB limit
    large_content = b"ID3" + b"\x00" * (2 * 1024 * 1024)  # 2MB
    mock_file = AsyncMock()
    mock_file.filename = "big.mp3"
    mock_file.read = AsyncMock(return_value=large_content)
    mock_file.seek = AsyncMock()
    with pytest.raises(FileTooLargeError):
        await validate_upload(mock_file, test_settings)


def test_cleanup_temp_removes_files(tmp_path):
    f = tmp_path / "test.wav"
    f.write_text("data")
    assert f.exists()
    cleanup_temp(f)
    assert not f.exists()


def test_cleanup_temp_handles_missing_files(tmp_path):
    f = tmp_path / "nonexistent.wav"
    cleanup_temp(f)  # Should not raise
