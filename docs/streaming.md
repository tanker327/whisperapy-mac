# Realtime Streaming Transcription

> **Endpoint:** `WS /api/v1/transcribe/stream`
> Realtime (low-latency) speech-to-text over a WebSocket, built on mlx-whisper.

---

## Table of Contents

1. [Overview](#1-overview)
2. [How It Works](#2-how-it-works)
3. [Wire Protocol](#3-wire-protocol)
4. [Message Reference](#4-message-reference)
5. [Configuration](#5-configuration)
6. [Audio Format Requirements](#6-audio-format-requirements)
7. [Examples](#7-examples)
8. [Tuning & Latency](#8-tuning--latency)
9. [Concurrency & Limits](#9-concurrency--limits)
10. [Error Handling](#10-error-handling)
11. [FAQ](#11-faq)

---

## 1. Overview

The streaming endpoint lets a client push audio **as it is captured** and receive
transcription results back incrementally, instead of uploading a complete file
and waiting for the whole transcript (as `POST /api/v1/transcribe` does).

It is exposed as a **WebSocket** at:

```
ws://<host>:<port>/api/v1/transcribe/stream
```

> **Note:** WebSocket endpoints do **not** appear in Swagger / `/docs`. The
> OpenAPI specification has no concept of WebSockets, so FastAPI omits this route
> from the schema. This is expected — the endpoint is fully functional, it just
> can't be documented or "tried out" from the Swagger UI. Use a WebSocket client
> (see [Examples](#7-examples)) instead.

### What "realtime" means here

mlx-whisper has **no incremental/streaming decode API** — it transcribes a whole
audio buffer and returns the full result. True token-by-token streaming (as in
purpose-built streaming ASR models) is therefore not possible with this model.

Instead, we approximate realtime by **re-transcribing a rolling window** of
audio. The result is "pseudo-realtime": you get partial results within a few
seconds of speaking, and they stabilize as more context arrives. This is the
same approach used by most Whisper-based "realtime" services.

---

## 2. How It Works

```
                         ┌──────────────────────────────────────────────┐
   binary PCM frames     │            StreamingSession                   │
  ───────────────────▶   │                                              │
                         │  buffer: [========= rolling window =========] │
   {"type":"flush"}      │             ▲committed▲   ▲  partial (tail) ▲ │
   {"type":"end"}        │             trimmed away    re-decoded each   │
  ───────────────────▶   │                              pass             │
                         └──────────────────────────────────────────────┘
        ▲                              │ every stream_window_seconds
        │                              ▼ of new audio
        │             TranscriberService.transcribe_array()  (asyncio.to_thread)
        │                              │
        │   JSON frames                ▼
        └──────────────  partial / final / done / error
```

The flow, step by step:

1. **Ingest.** Each binary frame of raw PCM is appended to an in-memory float32
   buffer. (A stray odd byte from a split 16-bit sample is held over to the next
   frame so samples never get corrupted.)

2. **Trigger.** Once `stream_window_seconds` of *new* audio has accumulated since
   the last pass, a transcription pass runs. Inference happens in a worker thread
   (`asyncio.to_thread`) so the event loop keeps receiving frames.

3. **Commit strategy.** mlx-whisper returns a list of segments for the current
   buffer. The **last** segment is treated as still-forming (it may change as
   more audio arrives) and emitted as a `partial`. All **earlier** segments are
   considered stable, emitted as `final`, and appended to the committed
   transcript.

4. **Trim.** The buffer is trimmed up to the start of the partial segment, so the
   next pass only re-decodes the unfinished tail plus new audio. This keeps each
   pass cheap and bounds memory.

5. **Context carry-over.** The committed text (last ~200 chars) is fed back into
   the next pass as `initial_prompt`, which improves continuity across windows.

6. **Language lock.** With auto-detect, the language detected on the first pass is
   locked for the rest of the session so later windows don't flip-flop.

7. **Finalize.** On `{"type":"end"}` the remaining buffer is transcribed with
   everything committed, and a single aggregate `done` message is sent.

> **Why timestamps are accurate across windows:** segments returned by
> mlx-whisper are relative to the *current buffer*. The session tracks a
> `base_offset` (absolute time of the buffer's first sample) and adds it to every
> segment, so all emitted timestamps are absolute from the start of the stream.

---

## 3. Wire Protocol

The connection is symmetric: the client sends a mix of **binary** (audio) and
**text** (JSON control) frames; the server sends **text** (JSON) frames.

### Client → Server

| Frame type | Payload | Meaning |
|-----------|---------|---------|
| **Binary** | raw 16-bit LE PCM, mono, 16 kHz | Audio data. Send as often as you like; chunk size is up to you. |
| **Text** | `{"type":"config","language":"en"}` | (Optional) Override language. Send **before** audio. `"auto"`/omitted = auto-detect. |
| **Text** | `{"type":"flush"}` | Force a transcription pass now (e.g. on a pause) without ending the session. |
| **Text** | `{"type":"end"}` | Finalize: transcribe the remainder, emit `done`, and close. |

### Server → Client

| `type` | Shape | Meaning |
|--------|-------|---------|
| `final` | `{"type":"final","segments":[{start,end,text}, ...]}` | Stabilized segments. Append these to your transcript. |
| `partial` | `{"type":"partial","segment":{start,end,text}}` | The current still-forming tail. **Replace** the previous partial with this. |
| `done` | `{"type":"done","language_detected","duration_seconds","text","segments"}` | Sent once after `end`. The full aggregate transcript. |
| `error` | `{"type":"error","message":"..."}` | Something went wrong (see [Error Handling](#10-error-handling)). |

### Typical sequence

```
client → {"type":"config","language":"en"}     (optional)
client → <binary PCM> <binary PCM> ...
server → {"type":"final","segments":[...]}      (as windows stabilize)
server → {"type":"partial","segment":{...}}
client → <binary PCM> ...
server → {"type":"final","segments":[...]}
server → {"type":"partial","segment":{...}}
client → {"type":"end"}
server → {"type":"final","segments":[...]}      (the tail, now committed)
server → {"type":"done", ...}
(connection closes)
```

> **Rendering tip:** keep a list of `final` segments plus a single "current"
> partial line. When a `partial` arrives, overwrite the current line; when
> `final` arrives, append it and clear the current line. This produces the
> familiar "live caption" effect.

---

## 4. Message Reference

### Segment object

Used inside `final.segments`, `partial.segment`, and `done.segments`:

```json
{ "start": 12.34, "end": 14.90, "text": "the transcribed words" }
```

- `start`, `end` — seconds, **absolute** from the start of the stream (rounded to ms).
- `text` — trimmed segment text.

### `done` object

```json
{
  "type": "done",
  "language_detected": "en",
  "duration_seconds": 42.18,
  "text": "full concatenated transcript ...",
  "segments": [ { "start": 0.0, "end": 3.1, "text": "..." }, ... ]
}
```

- `language_detected` — language used (detected on first pass, or your override).
  `null` if no audio was ever transcribed.
- `duration_seconds` — end time of the last committed segment, or `null`.
- `text` — all committed segment texts joined with spaces.
- `segments` — every committed segment.

---

## 5. Configuration

All settings live in `app/config.py` (Pydantic `BaseSettings`) and are
overridable via environment variables or `.env`:

| Setting | Env var | Default | Description |
|--------|---------|---------|-------------|
| `stream_sample_rate` | `STREAM_SAMPLE_RATE` | `16000` | Expected PCM sample rate (Hz). Whisper is trained on 16 kHz. |
| `stream_window_seconds` | `STREAM_WINDOW_SECONDS` | `5.0` | Run a pass after this much **new** audio. Lower = snappier partials, more compute. |
| `stream_max_buffer_seconds` | `STREAM_MAX_BUFFER_SECONDS` | `30.0` | Hard cap on buffer growth (Whisper's native 30 s window). Prevents unbounded memory if no segment boundaries appear. |

Example `.env`:

```dotenv
STREAM_WINDOW_SECONDS=3.0
STREAM_MAX_BUFFER_SECONDS=30.0
```

> Changing config requires a service restart (the dev server's `--reload` does
> not reliably pick up config/dependency changes; launchd has no reload).

---

## 6. Audio Format Requirements

The server consumes **raw PCM** — there is no server-side ffmpeg on the streaming
path (unlike the file/URL endpoints). The client is responsible for delivering:

| Property | Value |
|----------|-------|
| Encoding | 16-bit signed integer, **little-endian** (`s16le` / `pcm_s16le`) |
| Channels | **mono** (1 channel) |
| Sample rate | **16000 Hz** (must match `stream_sample_rate`) |
| Framing | any chunk size; the server handles odd byte boundaries |

To convert any file to this format with ffmpeg:

```bash
ffmpeg -i input.mp3 -vn -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -
```

For a live microphone on macOS (avfoundation), the capture side looks like:

```bash
ffmpeg -f avfoundation -i ":0" -f s16le -acodec pcm_s16le -ar 16000 -ac 1 -
```

---

## 7. Examples

### 7.1 Bundled client script

The repo ships `scripts/stream_client.py`, which decodes any media file with
ffmpeg, streams it at ~realtime pace, and prints results:

```bash
# Start the server first (make dev), then:
uv run python scripts/stream_client.py path/to/audio.mp3
uv run python scripts/stream_client.py path/to/audio.mp3 --language en
uv run python scripts/stream_client.py path/to/audio.mp3 --url ws://localhost:8000/api/v1/transcribe/stream
```

Output looks like:

```
  … the quick brown
[0.0-2.1] the quick brown fox
  … jumps over the lazy
[2.1-4.0] jumps over the lazy dog

=== DONE ===
the quick brown fox jumps over the lazy dog
```

### 7.2 Minimal Python client

```python
import asyncio, json, subprocess, websockets

URL = "ws://localhost:8000/api/v1/transcribe/stream"

async def main(path: str):
    # Decode to s16le/16k/mono via ffmpeg.
    ff = subprocess.Popen(
        ["ffmpeg", "-i", path, "-vn", "-f", "s16le",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    async with websockets.connect(URL) as ws:
        await ws.send(json.dumps({"type": "config", "language": "en"}))

        async def reader():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["type"] == "partial":
                    print("…", msg["segment"]["text"])
                elif msg["type"] == "final":
                    for s in msg["segments"]:
                        print(f"[{s['start']:.1f}] {s['text']}")
                elif msg["type"] == "done":
                    print("DONE:", msg["text"]); return
                elif msg["type"] == "error":
                    print("ERROR:", msg["message"]); return

        task = asyncio.create_task(reader())
        while chunk := ff.stdout.read(8000):   # ~0.25 s per frame
            await ws.send(chunk)
            await asyncio.sleep(0.25)           # pace to realtime
        await ws.send(json.dumps({"type": "end"}))
        await task

asyncio.run(main("audio.mp3"))
```

### 7.3 Browser (Web Audio + MediaRecorder is *not* enough)

Browsers don't natively produce raw 16 kHz mono PCM, so you must downsample in an
`AudioWorklet`/`ScriptProcessor` and convert Float32 → Int16 before sending.
Sketch:

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/transcribe/stream");
ws.binaryType = "arraybuffer";

ws.onopen = () => ws.send(JSON.stringify({ type: "config", language: "en" }));
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "partial") updateLiveLine(msg.segment.text);
  if (msg.type === "final")  msg.segments.forEach(s => appendLine(s.text));
  if (msg.type === "done")   console.log("done:", msg.text);
  if (msg.type === "error")  console.error(msg.message);
};

// In your audio callback, after downsampling `float32` to 16 kHz mono:
function sendPCM(float32) {
  const pcm = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  if (ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer);
}

function stop() { ws.send(JSON.stringify({ type: "end" })); }
```

> Downsampling from the browser's native 44.1/48 kHz to 16 kHz is required —
> sending the wrong sample rate produces garbled or chipmunk-speed transcripts.

### 7.4 Quick smoke test with `wscat`

```bash
npx wscat -c ws://localhost:8000/api/v1/transcribe/stream
> {"type":"end"}
< {"type":"done","language_detected":null,"duration_seconds":null,"text":"","segments":[]}
```

(`wscat` only sends text frames, so you can't feed audio with it — but it
confirms the endpoint is alive and the protocol responds.)

---

## 8. Tuning & Latency

**Perceived latency ≈ `stream_window_seconds` + inference time per window.**

- Lower `stream_window_seconds` (e.g. `2.0`–`3.0`) → faster, more frequent
  partials, but more re-transcription work (the tail is decoded repeatedly).
- Higher values → fewer passes, lower CPU/GPU load, but laggier captions.
- `whisper-large-v3-turbo` on Apple Silicon comfortably keeps up with realtime
  for short windows, so a 3–5 s window is a good default.

**Pacing:** the client should send audio at roughly realtime. Dumping an entire
file as fast as possible still works (you'll get results as windows fill), but it
won't behave like a live stream and will serialize a lot of inference at once.

---

## 9. Concurrency & Limits

- The mlx-whisper model is a **single shared singleton**. Inference is serialized
  across all active streams (and the sync endpoints). One or two concurrent
  streams are fine; many simultaneous streams will queue behind each other.
- Each connection keeps its audio buffer in memory, bounded by
  `stream_max_buffer_seconds`.
- For higher concurrency you'd need a worker pool / multiple model instances —
  out of scope for the current design.

---

## 10. Error Handling

WebSocket frames bypass the HTTP middleware and the global exception handlers
used by the REST endpoints. Errors on the stream are reported **in-band** as
JSON frames instead of HTTP status codes:

```json
{ "type": "error", "message": "Model not ready" }
```

Cases you may encounter:

| Situation | Behavior |
|-----------|----------|
| Model still loading at connect time | `{"type":"error","message":"Model not ready"}`, then close. Poll `GET /health` for `"model_loaded": true` first. |
| Invalid JSON in a text frame | `{"type":"error","message":"Invalid JSON control message"}`; connection stays open. |
| Unknown control `type` | `{"type":"error","message":"Unknown control type: ..."}`; connection stays open. |
| Transcription failure mid-stream | `{"type":"error","message":"..."}`, then the connection closes. |
| Client disconnects abruptly | Server logs it and cleans up the session; no further frames. |

> The server is defensive: an unexpected exception is logged, surfaced as an
> `error` frame when possible, and the socket is always closed in a `finally`
> block — a bad stream never crashes the server.

---

## 11. FAQ

**Q: Why isn't the endpoint in Swagger / `/docs`?**
WebSockets aren't part of the OpenAPI spec, so FastAPI can't document them there.
This is expected; the endpoint works regardless. Use a WS client to exercise it.

**Q: Can I send MP3/WAV bytes directly?**
No. The streaming path expects raw `s16le`/16 kHz/mono PCM. Decode client-side
(ffmpeg). For file/URL inputs in arbitrary formats, use `POST /api/v1/transcribe`
or `POST /api/v1/transcribe/url`, which run ffmpeg server-side.

**Q: Why do partial results sometimes change wording?**
That's by design — the partial is the still-forming tail being re-decoded with
more context each pass. Once enough audio follows, it stabilizes and is emitted
as `final`, after which it won't change.

**Q: How do I get the complete transcript?**
Either accumulate all `final` segments yourself, or send `{"type":"end"}` and
read the aggregate `text`/`segments` from the `done` message.

**Q: Do I have to send `end`?**
It's recommended — it flushes the final tail and gives you the `done` aggregate.
If the client just disconnects, already-committed segments were still delivered,
but the last in-progress tail won't be finalized.

---

## See Also

- `app/services/streaming.py` — `StreamingSession` (buffering, commit, trim logic)
- `app/services/transcriber.py` — `transcribe_array()` (in-memory inference)
- `app/api/v1/transcribe.py` — `transcribe_stream` (the WebSocket endpoint)
- `scripts/stream_client.py` — runnable example client / smoke test
- `tests/unit/test_streaming.py`, `tests/integration/test_stream.py` — behavior specs
