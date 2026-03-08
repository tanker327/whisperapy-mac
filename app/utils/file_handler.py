import re
import uuid
from pathlib import Path

from fastapi import UploadFile
from loguru import logger

from app.config import Settings
from app.core.exceptions import (
    FileTooLargeError,
    FileValidationError,
    UnsupportedFormatError,
)

# Magic bytes for supported formats
MAGIC_BYTES: dict[str, list[bytes]] = {
    "mp4": [
        b"\x00\x00\x00\x18ftypmp4",
        b"\x00\x00\x00\x1cftypmp4",
        b"\x00\x00\x00\x20ftypisom",
        b"\x00\x00\x00\x1cftypisom",
    ],
    "mov": [b"\x00\x00\x00\x14ftypqt"],
    "mkv": [b"\x1a\x45\xdf\xa3"],
    "webm": [b"\x1a\x45\xdf\xa3"],
    "avi": [b"RIFF"],
    "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
    "wav": [b"RIFF"],
    "m4a": [b"\x00\x00\x00\x20ftypM4A", b"\x00\x00\x00\x1cftypM4A"],
    "ogg": [b"OggS"],
    "flac": [b"fLaC"],
    "aac": [b"\xff\xf1", b"\xff\xf9"],
}


def sanitize_filename(filename: str) -> str:
    """Strip path separators and normalize characters."""
    name = Path(filename).name  # Strip directory components
    name = re.sub(r"[^\w\.\-]", "_", name)  # Replace unsafe chars
    return name


async def validate_upload(file: UploadFile, settings: Settings) -> None:
    """Validate file extension, magic bytes, and size."""
    if not file.filename:
        raise FileValidationError("No filename provided")

    # Check extension
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions:
        raise UnsupportedFormatError(f"File type .{ext} is not supported")

    # Check file size via content length or by reading
    content = await file.read()
    await file.seek(0)

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        size_mb = len(content) / 1024 / 1024
        raise FileTooLargeError(
            f"File size {size_mb:.1f}MB exceeds "
            f"limit of {settings.max_file_size_mb}MB"
        )

    # Check magic bytes
    _validate_magic_bytes(content, ext)


def _validate_magic_bytes(content: bytes, ext: str) -> None:
    """Verify file content matches expected magic bytes."""
    signatures = MAGIC_BYTES.get(ext)
    if not signatures:
        return  # No signature check for this format

    for sig in signatures:
        if content[: len(sig)] == sig:
            return

    raise UnsupportedFormatError(
        f"File content does not match expected format for .{ext}"
    )


async def save_temp_file(file: UploadFile, settings: Settings) -> Path:
    """Save upload to tmp/ with a sanitized, unique filename."""
    settings.temp_dir.mkdir(parents=True, exist_ok=True)

    original = sanitize_filename(file.filename or "upload")
    stem = Path(original).stem
    suffix = Path(original).suffix
    unique_name = f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
    dest = settings.temp_dir / unique_name

    content = await file.read()
    dest.write_bytes(content)
    await file.seek(0)

    logger.info(f"Saved temp file: {unique_name} ({len(content)} bytes)")
    return dest


def cleanup_temp(*paths: Path) -> None:
    """Remove temp files after transcription completes or fails."""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up: {path.name}")
        except OSError as e:
            logger.warning(f"Failed to clean up {path}: {e}")
