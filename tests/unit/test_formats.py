from app.schemas.transcription import Segment
from app.utils.formats import to_srt, to_vtt

SEGS = [
    Segment(start=0.0, end=1.5, text="Hello"),
    Segment(start=61.25, end=3661.999, text="World"),
]


def test_srt():
    assert to_srt(SEGS) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n"
        "2\n00:01:01,250 --> 01:01:01,999\nWorld\n"
    )


def test_vtt():
    assert to_vtt(SEGS) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.500\nHello\n\n"
        "00:01:01.250 --> 01:01:01.999\nWorld\n"
    )


def test_empty():
    assert to_srt([]) == ""
    assert to_vtt([]) == "WEBVTT\n"
