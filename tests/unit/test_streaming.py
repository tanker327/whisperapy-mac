from unittest.mock import MagicMock

import numpy as np
import pytest

from app.config import Settings
from app.services.streaming import StreamingSession


@pytest.fixture
def settings():
    return Settings(
        stream_sample_rate=16000,
        stream_window_seconds=1.0,
        stream_max_buffer_seconds=4.0,
        _env_file=None,
    )


def make_transcriber(results):
    """Transcriber stub whose transcribe_array returns each result in turn."""
    mock = MagicMock()
    mock.transcribe_array.side_effect = list(results)
    return mock


def pcm_bytes(seconds: float, sample_rate: int = 16000) -> bytes:
    return (np.zeros(int(seconds * sample_rate), dtype=np.int16)).tobytes()


def test_add_pcm_normalizes_and_counts(settings):
    session = StreamingSession(make_transcriber([]), settings)
    samples = np.array([0, 16384, -32768], dtype=np.int16)
    session.add_pcm(samples.tobytes())
    assert session._buffer.dtype == np.float32
    assert session._buffer[1] == pytest.approx(0.5, abs=1e-3)
    assert session._buffer[2] == pytest.approx(-1.0)
    assert session._received == 3


def test_add_pcm_handles_split_sample(settings):
    """An odd trailing byte is held over until the next frame completes it."""
    session = StreamingSession(make_transcriber([]), settings)
    raw = np.array([1000, 2000], dtype=np.int16).tobytes()  # 4 bytes
    session.add_pcm(raw[:3])  # one and a half samples
    assert session._received == 1
    assert session._leftover == raw[2:3]
    session.add_pcm(raw[3:])
    assert session._received == 2


async def test_should_process_after_one_window(settings):
    session = StreamingSession(make_transcriber([]), settings)
    session.add_pcm(pcm_bytes(0.5))
    assert session.should_process() is False
    session.add_pcm(pcm_bytes(0.6))  # now > 1.0s of new audio
    assert session.should_process() is True


async def test_process_commits_all_but_last_segment(settings):
    transcriber = make_transcriber(
        [
            {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello"},
                    {"start": 1.0, "end": 2.0, "text": "world"},
                ],
            }
        ]
    )
    session = StreamingSession(transcriber, settings)
    session.add_pcm(pcm_bytes(2.0))

    committed, partial = await session.process()

    assert [s["text"] for s in committed] == ["hello"]
    assert partial["text"] == "world"
    # Buffer trimmed up to the partial's start (1.0s) → base offset advanced.
    assert session._base_offset == pytest.approx(1.0)


async def test_process_final_commits_everything(settings):
    transcriber = make_transcriber(
        [
            {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "hello"},
                    {"start": 1.0, "end": 2.0, "text": "world"},
                ],
            }
        ]
    )
    session = StreamingSession(transcriber, settings)
    session.add_pcm(pcm_bytes(2.0))

    committed, partial = await session.process(final=True)

    assert [s["text"] for s in committed] == ["hello", "world"]
    assert partial is None
    summary = session.summary()
    assert summary["text"] == "hello world"
    assert summary["language_detected"] == "en"
    assert summary["duration_seconds"] == pytest.approx(2.0)


async def test_absolute_timestamps_accumulate_across_windows(settings):
    transcriber = make_transcriber(
        [
            {
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "one"},
                    {"start": 1.0, "end": 2.0, "text": "two"},
                ],
            },
            {
                "language": "en",
                # Times are relative to the trimmed buffer (which now starts at 1.0s).
                "segments": [
                    {"start": 0.0, "end": 1.5, "text": "two"},
                    {"start": 1.5, "end": 2.5, "text": "three"},
                ],
            },
        ]
    )
    session = StreamingSession(transcriber, settings)
    session.add_pcm(pcm_bytes(2.0))
    await session.process()

    session.add_pcm(pcm_bytes(1.0))
    committed, partial = await session.process()

    # "two" now stabilizes at an absolute offset (base 1.0 + rel 0.0).
    assert committed[0]["text"] == "two"
    assert committed[0]["start"] == pytest.approx(1.0)
    assert partial["text"] == "three"
    assert partial["start"] == pytest.approx(2.5)


async def test_locks_detected_language_for_later_windows(settings):
    transcriber = make_transcriber(
        [
            {"language": "fr", "segments": [{"start": 0.0, "end": 1.0, "text": "bon"}]},
            {"language": "fr", "segments": []},
        ]
    )
    session = StreamingSession(transcriber, settings, language="auto")
    session.add_pcm(pcm_bytes(2.0))
    await session.process()
    session.add_pcm(pcm_bytes(1.0))
    await session.process()

    # Second call should pass the locked-in language, not None/auto.
    assert transcriber.transcribe_array.call_args_list[1].kwargs["language"] == "fr"


async def test_oversized_single_segment_buffer_is_trimmed(settings):
    """A lone in-progress segment must not let the buffer grow unbounded."""
    transcriber = make_transcriber(
        [{"language": "en", "segments": [{"start": 0.0, "end": 2.0, "text": "long"}]}]
    )
    session = StreamingSession(transcriber, settings)
    session.add_pcm(pcm_bytes(5.0))  # exceeds 4.0s max buffer

    before = len(session._buffer)
    await session.process()
    assert len(session._buffer) < before
