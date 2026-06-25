#!/usr/bin/env python3
"""Example client for the realtime streaming endpoint.

Streams a media file to the WebSocket endpoint as raw 16kHz mono PCM and prints
partial/final segments as they arrive. ffmpeg does the decoding, so any input
format works. This doubles as a manual smoke test for the endpoint.

Usage:
    uv run python scripts/stream_client.py path/to/audio.mp3
    uv run python scripts/stream_client.py path/to/audio.mp3 --language en

For live microphone input, pipe ffmpeg's avfoundation capture into a similar
loop; the wire format is identical (s16le, mono, 16kHz).
"""

import argparse
import asyncio
import json
import subprocess

import websockets

SAMPLE_RATE = 16000
# ~0.25s of audio per frame (16000 * 2 bytes * 0.25).
CHUNK_BYTES = 8000


async def stream(path: str, url: str, language: str | None) -> None:
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-i", path,
            "-vn",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    async with websockets.connect(url) as ws:
        if language:
            await ws.send(json.dumps({"type": "config", "language": language}))

        async def receive() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "partial":
                    print(f"  … {msg['segment']['text']}", flush=True)
                elif kind == "final":
                    for seg in msg["segments"]:
                        print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
                elif kind == "done":
                    print("\n=== DONE ===")
                    print(msg["text"])
                    return
                elif kind == "error":
                    print(f"ERROR: {msg['message']}")
                    return

        receiver = asyncio.create_task(receive())

        # Feed audio at roughly realtime so partials behave like a live stream.
        while chunk := ffmpeg.stdout.read(CHUNK_BYTES):
            await ws.send(chunk)
            await asyncio.sleep(0.25)

        await ws.send(json.dumps({"type": "end"}))
        await receiver

    ffmpeg.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a media file")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/api/v1/transcribe/stream",
        help="WebSocket endpoint URL",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. en, fr). Omit for auto-detect.",
    )
    args = parser.parse_args()
    asyncio.run(stream(args.path, args.url, args.language))


if __name__ == "__main__":
    main()
