"""Upload validation, temp files, and guarded URL downloads.

Uploads are streamed to disk in chunks; the whole file is never held in
memory. URL downloads refuse loopback / private / link-local hosts unless
``ALLOW_PRIVATE_URLS`` is set, so the endpoint cannot be used as a proxy into
the network the server sits on.
"""

import asyncio
import ipaddress
import os
import re
import socket
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import UploadFile
from loguru import logger

from app.config import Settings
from app.core.exceptions import (
    DownloadError,
    FileTooLargeError,
    FileValidationError,
    ForbiddenUrlError,
    UnsupportedFormatError,
)

# Formats that use ISO base media file format (ftyp box at offset 4)
_FTYP_FORMATS = {"mp4", "mov", "m4a"}

# RIFF containers: "RIFF" + 4 size bytes + form type at offset 8
_RIFF_FORMATS = {"wav": b"WAVE", "avi": b"AVI "}

# Magic bytes for the remaining supported formats
MAGIC_BYTES: dict[str, list[bytes]] = {
    "mkv": [b"\x1a\x45\xdf\xa3"],
    "webm": [b"\x1a\x45\xdf\xa3"],
    "mp3": [b"ID3", b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3"],
    "ogg": [b"OggS"],
    "flac": [b"fLaC"],
    "aac": [b"\xff\xf1", b"\xff\xf9"],
}

_MAGIC_PEEK_BYTES = 16
_COPY_CHUNK = 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Strip path separators and normalize characters."""
    name = Path(filename).name  # Strip directory components
    name = re.sub(r"[^\w\.\-]", "_", name)  # Replace unsafe chars
    return name


# ---------------------------------------------------------------- validation


def validate_magic_bytes(head: bytes, ext: str) -> None:
    """Verify the first bytes of a file match the claimed extension."""
    if ext in _FTYP_FORMATS:
        if len(head) >= 8 and head[4:8] == b"ftyp":
            return
    elif ext in _RIFF_FORMATS:
        if head[:4] == b"RIFF" and head[8:12] == _RIFF_FORMATS[ext]:
            return
    else:
        signatures = MAGIC_BYTES.get(ext)
        if not signatures:
            return  # No signature check for this format
        if any(head.startswith(sig) for sig in signatures):
            return
    raise UnsupportedFormatError(
        f"File content does not match expected format for .{ext}"
    )


def upload_extension(file: UploadFile, settings: Settings) -> str:
    if not file.filename:
        raise FileValidationError("No filename provided")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions:
        raise UnsupportedFormatError(f"File type .{ext} is not supported")
    return ext


async def validate_upload(file: UploadFile, settings: Settings) -> None:
    """Validate extension, declared size, and magic bytes without reading the
    whole upload."""
    ext = upload_extension(file, settings)

    if file.size is not None and file.size > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {file.size / 1024 / 1024:.1f}MB exceeds "
            f"limit of {settings.max_file_size_mb}MB"
        )

    head = await file.read(_MAGIC_PEEK_BYTES)
    await file.seek(0)
    validate_magic_bytes(head, ext)


# ---------------------------------------------------------------- temp files


def _new_temp_path(settings: Settings, suffix: str = "") -> Path:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=settings.temp_dir, suffix=suffix)
    os.close(fd)
    return Path(name)


def _copy_limited(src, dest: Path, max_bytes: int) -> int:
    """Stream ``src`` (a binary file object) to ``dest``, enforcing a cap."""
    written = 0
    with open(dest, "wb") as out:
        while True:
            chunk = src.read(_COPY_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise FileTooLargeError(
                    f"File exceeds limit of {max_bytes // (1024 * 1024)}MB"
                )
            out.write(chunk)
    return written


async def save_temp_file(file: UploadFile, settings: Settings) -> Path:
    """Stream an upload to temp_dir with a sanitized, unique filename."""
    original = sanitize_filename(file.filename or "upload")
    stem, suffix = Path(original).stem, Path(original).suffix
    dest = _new_temp_path(settings, suffix=f"_{stem}{suffix}")

    await file.seek(0)
    try:
        written = await asyncio.to_thread(
            _copy_limited, file.file, dest, settings.max_file_size_bytes
        )
    except BaseException:
        cleanup_temp(dest)
        raise

    logger.info(f"Saved temp file: {dest.name} ({written} bytes)")
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


def sweep_temp_dir(settings: Settings) -> int:
    """Delete leftovers from crashed requests. Returns the number removed."""
    cutoff = time.time() - settings.temp_max_age_hours * 3600
    removed = 0
    if not settings.temp_dir.is_dir():
        return 0
    for path in settings.temp_dir.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError as e:
            logger.warning(f"Failed to sweep {path}: {e}")
    if removed:
        logger.info(f"Swept {removed} stale file(s) from {settings.temp_dir}")
    return removed


# ------------------------------------------------------------ URL downloads


def _is_public_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def check_url_allowed(url: str, settings: Settings) -> None:
    """Reject non-HTTP schemes and hosts that resolve to non-public addresses."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ForbiddenUrlError(f"URL scheme '{parts.scheme}' is not allowed")
    host = parts.hostname
    if not host:
        raise ForbiddenUrlError("URL has no host")
    if settings.allow_private_urls:
        return
    try:
        infos = socket.getaddrinfo(host, parts.port or 80, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise DownloadError(f"Could not resolve host '{host}'") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not _is_public_address(ip):
            raise ForbiddenUrlError(
                f"URL host '{host}' resolves to a non-public address ({ip})"
            )


async def download_file_from_url(url: str, settings: Settings) -> Path:
    """Stream-download a URL to a temp file, enforcing host rules and size."""
    await asyncio.to_thread(check_url_allowed, url, settings)
    max_bytes = settings.max_file_size_bytes
    dest = _new_temp_path(settings, suffix=f"_download_{uuid.uuid4().hex[:8]}")

    timeout = httpx.Timeout(
        connect=settings.download_connect_timeout,
        read=settings.download_read_timeout,
        write=60.0,
        pool=15.0,
    )

    async def on_redirect(response: httpx.Response) -> None:
        if response.next_request is not None:
            await asyncio.to_thread(
                check_url_allowed, str(response.next_request.url), settings
            )

    try:
        async with (
            httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                event_hooks={"response": [on_redirect]},
            ) as client,
            client.stream("GET", url) as response,
        ):
            if response.status_code >= 400:
                raise DownloadError(f"Download failed with HTTP {response.status_code}")

            downloaded = 0
            with open(dest, "wb") as f:  # noqa: ASYNC230 - 64 KB chunk writes
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise FileTooLargeError(
                            f"Download exceeds limit of {settings.max_file_size_mb}MB"
                        )
                    f.write(chunk)
    except (FileTooLargeError, DownloadError, ForbiddenUrlError):
        cleanup_temp(dest)
        raise
    except (httpx.HTTPError, OSError) as exc:
        cleanup_temp(dest)
        raise DownloadError(f"Download failed: {exc}") from exc

    logger.info(f"Downloaded {downloaded} bytes from URL to {dest.name}")
    return dest
