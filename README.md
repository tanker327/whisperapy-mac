# whisperapy-mac

Local REST transcription **and embedding** service powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) on Apple Silicon. No cloud, no usage fees, full privacy.

- **Mac native** — direct Metal GPU access via MLX (no Docker)
- **Fast** — `whisper-large-v3-turbo` runs ~8x faster than real time using ~1.5 GB of unified memory
- **Any format** — video and audio (mp4, mov, mkv, avi, webm, mp3, wav, m4a, ogg, flac, aac), decoded once with ffmpeg
- **OpenAI-compatible** — `/api/v1/audio/transcriptions`, `/api/v1/embeddings`, and `/api/v1/models` work with the OpenAI SDK by changing `base_url`
- **Embeddings** — `Qwen3-Embedding-4B-4bit-DWQ` with configurable token limits, batching, `base64` output, and Matryoshka `dimensions`
- **Production-grade** — streaming uploads, one-job GPU gate with honest `Retry-After`, optional API key, SSRF-guarded URL fetches, structured logging with rotation, request tracing

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (`brew install ffmpeg`)

## Quick Start

```bash
make install-dev          # uv sync --extra dev
cp .env.example .env      # optional: every value in it is already the default
make dev                  # first run downloads whisper (~1.5 GB) + Qwen3 (~2 GB)
```

The server listens on `http://localhost:8000`. Interactive docs at `/docs`. `make dev` honours `HOST` and `PORT` from the environment.

## API

All routes below live under `/api/v1`. When `API_KEY` is set they require `Authorization: Bearer <key>` or `X-API-Key: <key>`; the health routes never do.

### Health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.1.0",
  "model": "whisper-large-v3-turbo",
  "model_loaded": true,
  "embedding_model": "Qwen3-Embedding-4B-4bit-DWQ",
  "embedding_model_loaded": true,
  "uptime_seconds": 3600,
  "busy": false,
  "active_jobs": 0,
  "queued_jobs": 0,
  "max_concurrent_jobs": 1,
  "max_queued_jobs": 1,
  "estimated_wait_seconds": null
}
```

`/health` returns **503** with `"status": "starting"` until both models are loaded, so monitors can tell booting from broken. `/health/model` reports the configured repos and limits.

### Transcribe an upload

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@recording.mp4" \
  -F "language=auto" \
  -F "include_segments=true"
```

```json
{
  "job_id": "abc-123",
  "language_detected": "en",
  "duration_seconds": 124.5,
  "processing_time_seconds": 8.2,
  "text": "Full transcript here...",
  "segments": [
    { "start": 0.0, "end": 3.2, "text": "Hello world", "words": [] },
    { "start": 3.2, "end": 6.1, "text": "How are you?", "words": [] }
  ]
}
```

**Form fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Video or audio file |
| `language` | string | `auto` | Language code (`en`, `zh`, ...) or `auto` to detect |
| `include_segments` | bool | `false` | Include segment timings in JSON output |
| `word_timestamps` | bool | `false` | Populate `segments[].words` with per-word timings |
| `initial_prompt` | string | – | Vocabulary / context hint for the first window |
| `temperature` | float 0–1 | Whisper fallback ladder | Fix a single decoding temperature |
| `condition_on_previous_text` | bool | `true` | Feed prior text as context to the next window |
| `output_format` | enum | `json` | `json`, `verbose_json` (always with segments), `text`, `srt`, `vtt` |

### Transcribe from a URL

```bash
curl -X POST http://localhost:8000/api/v1/transcribe/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/talk.mp3", "output_format": "srt"}'
```

Accepts the same options as the upload endpoint as JSON fields. Only `http`/`https` URLs to **public** addresses are fetched; loopback, private, and link-local hosts are refused (also after redirects) unless `ALLOW_PRIVATE_URLS=true`. The busy check runs **before** the download starts.

### OpenAI-compatible transcription

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api/v1", api_key="unused-or-your-API_KEY"
)
with open("recording.mp3", "rb") as f:
    result = client.audio.transcriptions.create(
        model="whisper-1",  # accepted; the server's model is used
        file=f,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
```

`POST /api/v1/audio/transcriptions` accepts `file`, `model`, `language`, `prompt`, `response_format` (`json`, `verbose_json`, `text`, `srt`, `vtt`), `temperature`, and `timestamp_granularities[]`.

### Embeddings

```bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "goodbye"], "dimensions": 1024}'
```

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.02, "..."] },
    { "object": "embedding", "index": 1, "embedding": [0.03, -0.04, "..."] }
  ],
  "model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 },
  "truncated": 0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string \| string[] | required | Up to `EMBEDDING_MAX_BATCH` texts |
| `model` | string | – | Accepted for compatibility; the server model is authoritative |
| `encoding_format` | `float` \| `base64` | `float` | `base64` returns little-endian float32, as OpenAI does |
| `dimensions` | int 32–4096 | – | Matryoshka truncation with re-normalisation |
| `prompt` | string | – | Instruction prepended to each input (Qwen3 query-side embeddings) |

Inputs longer than `EMBEDDING_MAX_TOKENS` return **422** by default. With `EMBEDDING_TRUNCATE=true` they are cut instead and the response's `truncated` count says how many. `usage` always reflects the tokens actually embedded.

`GET /api/v1/models` lists both models for SDKs that probe it.

### Errors

Every error uses one shape and never includes a stack trace:

```json
{ "error": "UnsupportedFormatError", "message": "File type .exe is not supported", "request_id": "…" }
```

| Status | Errors |
|--------|--------|
| 401 | `AuthenticationError` |
| 413 | `FileTooLargeError` |
| 415 | `UnsupportedFormatError` |
| 422 | `FileValidationError`, `DownloadError`, `ForbiddenUrlError`, `InvalidRequestError`, `InputTooLongError`, request validation |
| 500 | `AudioExtractionError`, `TranscriptionError`, `EmbeddingError`, `InternalServerError` |
| 503 | `ModelNotReadyError`, `ServiceBusyError` (with `Retry-After` header and `retry_after_seconds`) |

Every response carries `X-Request-ID` (an incoming one is honoured) and `X-Processing-Time`.

## Concurrency and busy responses

The Metal GPU runs one model call at a time, so transcription and embedding requests share a single job slot. Both models are loaded on, and every model call runs on, one dedicated MLX thread (MLX keeps GPU streams per thread). ffmpeg and tokenization run in ordinary threads. The event loop is never blocked, so `/health` and busy rejections answer instantly during a long transcription.

When a request arrives while the GPU is busy:

- If the running job is expected to finish within `QUEUE_WAIT_SECONDS`, the request waits for the slot (at most `MAX_QUEUED_JOBS` may wait).
- Otherwise it fails immediately with **503**, a `Retry-After` header, and `retry_after_seconds` in the body.

The estimate uses the decoded audio length and `TRANSCRIBE_SPEED_FACTOR` for transcription, and the token count and `EMBED_TOKENS_PER_SECOND` for embeddings.

Note that FastAPI receives a multipart upload before the handler runs, so a busy rejection on the upload route still costs the client its upload. Poll `/health` first if that matters; the URL route rejects before downloading.

## Configuration

Everything is an environment variable or a `.env` entry. See [`.env.example`](.env.example); the values there are the defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Debug logging |
| `API_KEY` | – | Require a key on `/api/v1` when set |
| `CORS_ORIGINS` | `[]` | JSON list of allowed browser origins; empty disables CORS |
| `ALLOW_PRIVATE_URLS` | `false` | Let the URL endpoint fetch loopback / LAN addresses |
| `LOG_FORMAT` | `text` | `text` or `json` |
| `LOG_FILE` | – | Rotating log file path (`LOG_ROTATION`, `LOG_RETENTION`) |
| `MODEL_REPO` | `mlx-community/whisper-large-v3-turbo` | Transcription model |
| `EMBEDDING_MODEL_REPO` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | Embedding model |
| `EMBEDDING_MAX_TOKENS` | `8192` | Per-input token limit |
| `EMBEDDING_TRUNCATE` | `false` | Cut over-long inputs instead of rejecting |
| `EMBEDDING_MAX_BATCH` | `2048` | Max strings per request |
| `EMBEDDING_BATCH_SIZE` | `32` | Strings per GPU forward pass |
| `MAX_QUEUED_JOBS` | `1` | Requests allowed to wait for the GPU |
| `QUEUE_WAIT_SECONDS` | `15` | Max wait before 503 |
| `TRANSCRIBE_SPEED_FACTOR` | `8` | Assumed speed vs. real time for `Retry-After` |
| `EMBED_TOKENS_PER_SECOND` | `2000` | Assumed embedding throughput for `Retry-After` |
| `MAX_FILE_SIZE_MB` | `1500` | Upload / download limit |
| `TEMP_DIR` | `/tmp/whisperapy` | Working directory for media files |
| `TEMP_MAX_AGE_HOURS` | `6` | Stale temp files older than this are swept at startup |
| `FFMPEG_TIMEOUT_SECONDS` | `120` | ffmpeg timeout floor, plus `FFMPEG_TIMEOUT_SECONDS_PER_MB` (`0.5`) per MB |
| `DOWNLOAD_CONNECT_TIMEOUT` / `DOWNLOAD_READ_TIMEOUT` | `15` / `120` | httpx timeouts for URL downloads |

The version is read from package metadata (`pyproject.toml`) and is not configurable.

## Development

```bash
make test        # pytest with coverage (fails under 90%)
make lint        # ruff check + ruff format --check
make format      # ruff --fix + ruff format
make typecheck   # pyright
make check       # lint + typecheck + test
make clean       # temp files and tool caches
```

Tests mock mlx-whisper, mlx-embeddings, ffmpeg, and the network, so they run on any machine in about two seconds. CI runs the same `make check` on a macOS runner. `pre-commit install` enables the ruff hooks.

See [`docs/prod-deploy.md`](docs/prod-deploy.md) for running under launchd and [`CLAUDE.md`](CLAUDE.md) for the architecture notes.
