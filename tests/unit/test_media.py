import subprocess
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import Settings
from app.core.exceptions import AudioExtractionError
from app.services.media import SAMPLE_RATE, MediaService, load_wav


def write_wav(path: Path, seconds: float, rate: int = SAMPLE_RATE) -> None:
    pcm = (np.linspace(-1, 1, int(seconds * rate)) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def test_load_wav_returns_float32_with_exact_duration(tmp_path):
    wav = tmp_path / "a.wav"
    write_wav(wav, 2.5)
    audio = load_wav(wav)
    assert audio.samples.dtype == np.float32
    assert audio.duration_seconds == 2.5
    assert float(audio.samples.min()) >= -1.0 and float(audio.samples.max()) <= 1.0


def test_load_wav_rejects_garbage(tmp_path):
    bad = tmp_path / "a.wav"
    bad.write_bytes(b"not a wav")
    with pytest.raises(AudioExtractionError):
        load_wav(bad)


def test_load_wav_missing_file(tmp_path):
    with pytest.raises(AudioExtractionError):
        load_wav(tmp_path / "missing.wav")


@pytest.fixture
def media(tmp_path):
    return MediaService(
        Settings(
            temp_dir=tmp_path,
            ffmpeg_timeout_seconds=10,
            ffmpeg_timeout_seconds_per_mb=1,
            _env_file=None,
        )
    )


def test_extract_audio_builds_command_and_scales_timeout(media, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB -> 10 + 2 = 12s
    out = tmp_path / "out.wav"
    ok = MagicMock(returncode=0, stderr="")
    with patch("app.services.media.subprocess.run", return_value=ok) as run:
        assert media.extract_audio(src, out) == out
    cmd = run.call_args[0][0]
    assert cmd[0] == "ffmpeg" and "-nostdin" in cmd
    assert cmd[cmd.index("-ar") + 1] == str(SAMPLE_RATE)
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[-1] == str(out)
    assert run.call_args[1]["timeout"] == 12


def test_extract_audio_nonzero_exit(media, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    failed = MagicMock(returncode=1, stderr="Invalid data found")
    with (
        patch("app.services.media.subprocess.run", return_value=failed),
        pytest.raises(AudioExtractionError, match="Invalid data"),
    ):
        media.extract_audio(src, tmp_path / "out.wav")


def test_extract_audio_timeout(media, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    with (
        patch(
            "app.services.media.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
        ),
        pytest.raises(AudioExtractionError, match="timed out"),
    ):
        media.extract_audio(src, tmp_path / "out.wav")


def test_extract_audio_ffmpeg_missing(media, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    with (
        patch("app.services.media.subprocess.run", side_effect=FileNotFoundError()),
        pytest.raises(AudioExtractionError, match="not installed"),
    ):
        media.extract_audio(src, tmp_path / "out.wav")


def test_extract_and_load_decodes_once(media, tmp_path):
    """One ffmpeg call, then the WAV is read directly (no second decode)."""
    src = tmp_path / "in.mp3"
    src.write_bytes(b"x")
    out = tmp_path / "out.wav"

    def fake_run(cmd, **kwargs):
        write_wav(out, 1.0)
        return MagicMock(returncode=0, stderr="")

    with patch("app.services.media.subprocess.run", side_effect=fake_run) as run:
        audio = media.extract_and_load(src, out)
    assert run.call_count == 1
    assert audio.duration_seconds == 1.0


def test_load_wav_rejects_stereo(tmp_path):
    stereo = tmp_path / "s.wav"
    with wave.open(str(stereo), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"\x00" * 64)
    with pytest.raises(AudioExtractionError, match="16-bit mono"):
        load_wav(stereo)


def test_extract_audio_missing_input_uses_timeout_floor(media, tmp_path):
    ok = MagicMock(returncode=0, stderr="")
    with patch("app.services.media.subprocess.run", return_value=ok) as run:
        media.extract_audio(tmp_path / "missing.mp4", tmp_path / "out.wav")
    assert run.call_args[1]["timeout"] == 10  # floor only, size unknown


def test_extract_audio_generic_oserror(media, tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    with (
        patch("app.services.media.subprocess.run", side_effect=PermissionError("no")),
        pytest.raises(AudioExtractionError, match="Audio extraction failed"),
    ):
        media.extract_audio(src, tmp_path / "out.wav")
