import io
import os
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.exceptions import (
    DownloadError,
    FileTooLargeError,
    FileValidationError,
    ForbiddenUrlError,
    UnsupportedFormatError,
)
from app.utils.file_handler import (
    check_url_allowed,
    cleanup_temp,
    download_file_from_url,
    sanitize_filename,
    save_temp_file,
    sweep_temp_dir,
    validate_magic_bytes,
    validate_upload,
)

# --------------------------------------------------------------- filenames


def test_sanitize_filename_strips_path():
    assert sanitize_filename("/etc/passwd") == "passwd"
    assert sanitize_filename("../../secret.txt") == "secret.txt"


def test_sanitize_filename_replaces_unsafe_chars():
    result = sanitize_filename("my file (1).mp3")
    assert " " not in result and "(" not in result
    assert result.endswith(".mp3")


# ------------------------------------------------------------- magic bytes


@pytest.mark.parametrize(
    "ext,head",
    [
        ("mp4", b"\x00\x00\x00\x18ftypisom\x00\x00"),
        ("mov", b"\x00\x00\x00\x14ftypqt  "),
        ("m4a", b"\x00\x00\x00\x1cftypM4A "),
        ("wav", b"RIFF\x24\x08\x00\x00WAVEfmt "),
        ("avi", b"RIFF\x24\x08\x00\x00AVI LIST"),
        ("mp3", b"ID3\x04\x00"),
        ("mp3", b"\xff\xfb\x90\x00"),
        ("mp3", b"\xff\xfa\x90\x00"),
        ("mp3", b"\xff\xe3\x90\x00"),
        ("mkv", b"\x1a\x45\xdf\xa3"),
        ("webm", b"\x1a\x45\xdf\xa3"),
        ("ogg", b"OggS\x00"),
        ("flac", b"fLaC\x00"),
        ("aac", b"\xff\xf1\x50\x80"),
    ],
)
def test_validate_magic_bytes_accepts(ext, head):
    validate_magic_bytes(head, ext)


@pytest.mark.parametrize(
    "ext,head",
    [
        ("mp4", b"\x00" * 16),
        ("mp4", b"\x00\x00"),  # too short for ftyp
        ("wav", b"RIFF\x24\x08\x00\x00AVI LIST"),  # AVI renamed to .wav
        ("avi", b"RIFF\x24\x08\x00\x00WAVEfmt "),
        ("mp3", b"\x00\x00\x00\x00"),
        ("flac", b"OggS"),
    ],
)
def test_validate_magic_bytes_rejects(ext, head):
    with pytest.raises(UnsupportedFormatError):
        validate_magic_bytes(head, ext)


def test_validate_magic_bytes_unknown_ext_passes():
    validate_magic_bytes(b"anything", "opus")


# --------------------------------------------------------- validate_upload


def upload(filename, content: bytes, size=None):
    f = AsyncMock()
    f.filename = filename
    f.size = len(content) if size is None else size
    f.read = AsyncMock(return_value=content[:16])
    f.seek = AsyncMock()
    f.file = io.BytesIO(content)
    return f


async def test_validate_upload_no_filename(test_settings):
    with pytest.raises(FileValidationError):
        await validate_upload(upload(None, b""), test_settings)


async def test_validate_upload_bad_extension(test_settings):
    with pytest.raises(UnsupportedFormatError):
        await validate_upload(upload("file.exe", b"\x00" * 100), test_settings)


async def test_validate_upload_too_large_by_declared_size(test_settings):
    test_settings.max_file_size_mb = 1
    f = upload("big.mp3", b"ID3" + b"\x00" * 13, size=2 * 1024 * 1024)
    with pytest.raises(FileTooLargeError):
        await validate_upload(f, test_settings)
    f.read.assert_not_called()  # rejected before touching the body


async def test_validate_upload_reads_only_a_peek(test_settings):
    f = upload("ok.mp3", b"ID3" + b"\x00" * 5000)
    await validate_upload(f, test_settings)
    f.read.assert_awaited_once_with(16)
    f.seek.assert_awaited_with(0)


async def test_validate_upload_magic_mismatch(test_settings):
    with pytest.raises(UnsupportedFormatError):
        await validate_upload(upload("fake.mp3", b"\x00" * 100), test_settings)


# ---------------------------------------------------------- save_temp_file


async def test_save_temp_file_streams_to_disk(test_settings):
    content = b"ID3" + os.urandom(3 * 1024 * 1024)  # 3 MB, several chunks
    dest = await save_temp_file(upload("clip.mp3", content), test_settings)
    try:
        assert dest.parent == test_settings.temp_dir
        assert dest.name.endswith("_clip.mp3")
        assert dest.read_bytes() == content
    finally:
        dest.unlink()


async def test_save_temp_file_enforces_limit_and_cleans_up(test_settings):
    test_settings.max_file_size_mb = 1
    content = b"ID3" + b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(FileTooLargeError):
        await save_temp_file(upload("big.mp3", content), test_settings)
    assert list(test_settings.temp_dir.iterdir()) == []


async def test_save_temp_file_unique_names(test_settings):
    a = await save_temp_file(upload("x.mp3", b"ID3aaa"), test_settings)
    b = await save_temp_file(upload("x.mp3", b"ID3bbb"), test_settings)
    assert a != b
    cleanup_temp(a, b)


# ---------------------------------------------------------------- cleanup


def test_cleanup_temp_removes_files(tmp_path):
    f = tmp_path / "test.wav"
    f.write_text("data")
    cleanup_temp(f)
    assert not f.exists()


def test_cleanup_temp_handles_missing_files(tmp_path):
    cleanup_temp(tmp_path / "nonexistent.wav")


def test_sweep_temp_dir_removes_only_stale_files(test_settings):
    test_settings.temp_dir.mkdir(parents=True)
    test_settings.temp_max_age_hours = 1
    old = test_settings.temp_dir / "old.wav"
    new = test_settings.temp_dir / "new.wav"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    stale = time.time() - 2 * 3600
    os.utime(old, (stale, stale))
    assert sweep_temp_dir(test_settings) == 1
    assert not old.exists() and new.exists()


def test_sweep_temp_dir_missing_dir(test_settings):
    assert sweep_temp_dir(test_settings) == 0


# ---------------------------------------------------------- URL guarding


def addrinfo(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


@pytest.mark.parametrize("url", ["ftp://example.com/a", "file:///etc/passwd"])
def test_check_url_rejects_non_http(url, test_settings):
    with pytest.raises(ForbiddenUrlError):
        check_url_allowed(url, test_settings)


@pytest.mark.parametrize(
    "ip", ["127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254"]
)
def test_check_url_rejects_private_addresses(ip, test_settings):
    with (
        patch("app.utils.file_handler.socket.getaddrinfo", return_value=addrinfo(ip)),
        pytest.raises(ForbiddenUrlError),
    ):
        check_url_allowed("http://internal.example/x", test_settings)


def test_check_url_allows_public_address(test_settings):
    with patch(
        "app.utils.file_handler.socket.getaddrinfo",
        return_value=addrinfo("93.184.216.34"),
    ):
        check_url_allowed("https://example.com/x", test_settings)


def test_check_url_private_allowed_by_setting(test_settings):
    test_settings.allow_private_urls = True
    with patch("app.utils.file_handler.socket.getaddrinfo") as gai:
        check_url_allowed("http://127.0.0.1:8000/x", test_settings)
    gai.assert_not_called()


def test_check_url_unresolvable_host(test_settings):
    with (
        patch(
            "app.utils.file_handler.socket.getaddrinfo",
            side_effect=socket.gaierror("nope"),
        ),
        pytest.raises(DownloadError, match="resolve"),
    ):
        check_url_allowed("http://nope.invalid/x", test_settings)


# --------------------------------------------------------------- download


class FakeResponse:
    def __init__(self, status_code=200, chunks=None):
        self.status_code = status_code
        self._chunks = chunks or [b"fake audio data"]

    async def aiter_bytes(self, chunk_size=None):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.kwargs = None

    def stream(self, method, url):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def public_dns():
    with patch(
        "app.utils.file_handler.socket.getaddrinfo",
        return_value=addrinfo("93.184.216.34"),
    ):
        yield


async def test_download_success(test_settings, public_dns):
    fake_client = FakeClient(FakeResponse(chunks=[b"hello ", b"world"]))
    with patch("app.utils.file_handler.httpx.AsyncClient", return_value=fake_client):
        path = await download_file_from_url(
            "http://example.com/file.mp3", test_settings
        )
    assert path.parent == test_settings.temp_dir
    assert path.read_bytes() == b"hello world"
    path.unlink()


async def test_download_http_error_cleans_up(test_settings, public_dns):
    fake_client = FakeClient(FakeResponse(status_code=404))
    with (
        patch("app.utils.file_handler.httpx.AsyncClient", return_value=fake_client),
        pytest.raises(DownloadError, match="HTTP 404"),
    ):
        await download_file_from_url("http://example.com/missing", test_settings)
    assert list(test_settings.temp_dir.iterdir()) == []


async def test_download_size_exceeded_cleans_up(test_settings, public_dns):
    test_settings.max_file_size_mb = 1
    fake_client = FakeClient(FakeResponse(chunks=[b"\x00" * (2 * 1024 * 1024)]))
    with (
        patch("app.utils.file_handler.httpx.AsyncClient", return_value=fake_client),
        pytest.raises(FileTooLargeError),
    ):
        await download_file_from_url("http://example.com/big", test_settings)
    assert list(test_settings.temp_dir.iterdir()) == []


async def test_download_private_host_never_opens_connection(test_settings):
    client_cls = MagicMock()
    with (
        patch(
            "app.utils.file_handler.socket.getaddrinfo",
            return_value=addrinfo("10.1.1.1"),
        ),
        patch("app.utils.file_handler.httpx.AsyncClient", client_cls),
        pytest.raises(ForbiddenUrlError),
    ):
        await download_file_from_url("http://intranet/secret.mp3", test_settings)
    client_cls.assert_not_called()


async def test_download_transport_error(test_settings, public_dns):
    import httpx

    class Boom(FakeClient):
        def stream(self, method, url):
            raise httpx.ConnectError("refused")

    with (
        patch("app.utils.file_handler.httpx.AsyncClient", return_value=Boom(None)),
        pytest.raises(DownloadError, match="refused"),
    ):
        await download_file_from_url("http://example.com/x", test_settings)
