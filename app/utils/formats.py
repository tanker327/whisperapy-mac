"""Render a transcript as plain text, SRT, or WebVTT."""

from app.schemas.transcription import Segment


def _timestamp(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(segments: list[Segment]) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _timestamp(seg.start, ",")
        end = _timestamp(seg.end, ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)
