from pathlib import Path

from app.services.media import wav_duration_seconds


def test_wav_duration_from_size(tmp_path: Path):
    wav = tmp_path / "a.wav"
    with open(wav, "wb") as f:
        f.truncate(44 + 10 * 32000)  # header + 10s of 16kHz mono 16-bit
    assert wav_duration_seconds(wav) == 10.0


def test_wav_duration_header_only_is_zero(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x" * 44)
    assert wav_duration_seconds(wav) == 0.0


def test_wav_duration_missing_file_is_none(tmp_path: Path):
    assert wav_duration_seconds(tmp_path / "missing.wav") is None
