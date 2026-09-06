# whisperapy-mac

> **Mac Native · FastAPI · mlx-whisper · Apple Silicon M4**
> Project Design Document — Version 1.1

---

> **Status (v1.1, September 2026).** This document describes the original v1
> design. The code review in September 2026 kept the architecture and changed
> the following; where this document disagrees with `README.md` or
> `CLAUDE.md`, those files are authoritative.
>
> - **Uploads stream to disk** (16-byte magic peek, chunked copy with a size
>   cap). The whole file is never held in memory.
> - **Audio is decoded once.** ffmpeg produces a WAV which is loaded into a
>   numpy array and passed to `mlx_whisper.transcribe`; duration is exact.
> - **Warm-up is real**: one second of silence, failures propagate and abort
>   startup. `/health` returns 503 `starting` until both models are loaded.
> - **Embeddings**: `EMBEDDING_MAX_TOKENS` (422 or flagged truncation),
>   `EMBEDDING_MAX_BATCH`, chunked GPU batches, `base64`, `dimensions`,
>   `GET /api/v1/models`. Token counting runs off the MLX thread and feeds the
>   gate's estimate (`estimated_seconds` replaced `audio_seconds`).
> - **Security**: optional `API_KEY`, SSRF guard on URL fetches (public hosts
>   only, re-checked on redirects), CORS from settings.
> - **Transcription API**: `word_timestamps`, `initial_prompt`, `temperature`,
>   `condition_on_previous_text`, `output_format` (json / verbose_json / text /
>   srt / vtt), and an OpenAI-compatible `POST /api/v1/audio/transcriptions`.
>   `JobStatus` was removed (there are no async jobs).
> - **Structure**: exceptions carry their own `status_code`; one pure-ASGI
>   middleware replaces two `BaseHTTPMiddleware`s; health routes live in
>   `app/api/health.py`; `Annotated` dependency aliases; rotating/JSON logs.
> - **Tooling**: ruff replaces black, pyright in `make check`, coverage gate,
>   GitHub Actions CI, pre-commit, `ffmpeg-python` removed, version read from
>   package metadata.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Folder Structure](#3-folder-structure)
4. [Configuration](#4-configuration)
5. [API Endpoints](#5-api-endpoints)
6. [Processing Pipeline](#6-processing-pipeline)
7. [Core Layer](#7-core-layer)
8. [Services](#8-services)
9. [Testing Strategy](#9-testing-strategy)
10. [Makefile Commands](#10-makefile-commands)
11. [Best Practices Checklist](#11-best-practices-checklist)
12. [Implementation Order](#12-implementation-order)

---

## 1. Project Overview

**whisperapy-mac** is a local REST service that accepts any video or audio file and returns a high-quality transcription. It is designed to run natively on Apple Silicon (M4), leveraging Metal GPU acceleration for real-time transcription performance.

> **Goal:** Build a fast, accurate, local transcription REST service — no cloud required, no usage fees, full privacy.

### Key Characteristics

- Mac native — no Docker, no VMs, full Metal GPU access
- Models loaded at startup — no per-request load cost
- Supports any video or audio format via ffmpeg
- Sync and async transcription endpoints
- OpenAI-compatible text-embedding endpoint (Qwen3-Embedding via mlx-embeddings)
- Multiple output formats: JSON, plain text, SRT, VTT
- Production-grade: structured logging, error handling, request tracing

### Why Mac Native?

The `mlx-whisper` library uses Apple's MLX framework to run directly on Apple Silicon's Neural Engine and GPU via Metal. Docker on macOS runs inside a Linux VM which cannot access Metal.

| Approach | GPU Access | Speed | Use Case |
|---|---|---|---|
| Mac Native (this project) | Full Metal | Real-time+ | Best performance |
| Docker on Mac | None (VM) | CPU only | Portability testing |
| Linux Server | CUDA (if available) | Variable | Remote deployment |

---

## 2. Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Platform | macOS (Apple Silicon M4) | Metal GPU, unified memory |
| Package Manager | `uv` | Fast, modern Python dep management |
| Web Framework | `FastAPI` | Async, auto Swagger docs, file uploads |
| ASGI Server | `uvicorn` | Production-grade ASGI server |
| Transcription | `mlx-whisper` | Metal-accelerated on Apple Silicon |
| Model | `whisper-large-v3-turbo` | Best speed/quality balance |
| Embeddings | `mlx-embeddings` | Metal-accelerated text embeddings |
| Embedding Model | `Qwen3-Embedding-4B-4bit-DWQ` | High-quality 2560-dim vectors |
| Audio Extraction | `ffmpeg` via `subprocess` + `numpy` | Any container → 16 kHz mono WAV, loaded once into a float32 array |
| File Uploads | `python-multipart` | Required by FastAPI for multipart/form-data |
| Settings | `Pydantic BaseSettings` | Type-safe env config |
| Validation | `Pydantic v2` | Request/response schemas |
| Logging | `Loguru` | Structured, easy async logging |
| Formatting + Linting | `Ruff` | `ruff format` (black-compatible) + rules E F I B UP SIM ASYNC RUF |
| Type checking | `pyright` | Standard mode on `app/`, part of `make check` |
| Testing | `pytest` + `pytest-asyncio` + `pytest-cov` + `httpx` | Async test client, 90% coverage gate |
| CI | GitHub Actions (`macos-14`) | `make check` on every push / PR |
| System Dep | `ffmpeg` (Homebrew) | `brew install ffmpeg` |

### Model Selection

`whisper-large-v3-turbo` was released by OpenAI in late 2024 specifically to maximize speed without sacrificing accuracy.

| Model | Speed vs Base | Quality | Memory |
|---|---|---|---|
| `large-v3` (baseline) | 1x | ★★★★★ | ~3 GB |
| **`large-v3-turbo` (chosen)** | **~8x faster** | **★★★★★** | **~1.5 GB** |
| `distil-large-v3` | ~6x faster | ★★★★ | ~1.5 GB |
| `medium` | ~5x faster | ★★★ | ~1.5 GB |

> **M4 Headroom:** `large-v3-turbo` uses only ~1.5 GB of the M4's 24 GB unified memory — leaving plenty of capacity to run other models or services simultaneously.

---

## 3. Folder Structure

```
whisperapy-mac/
├── pyproject.toml              # uv + ruff + pyright + pytest config + all dependencies
├── uv.lock                     # Pinned dependency versions
├── .python-version             # Pin Python 3.12
├── .env                        # Local env vars (gitignored)
├── .env.example                # Every setting with its default
├── .gitignore
├── .pre-commit-config.yaml     # ruff hooks
├── .github/workflows/ci.yml    # make check on macos-14
├── Makefile                    # dev / test / lint / format / typecheck / check / clean
├── README.md
├── CLAUDE.md                   # Architecture notes for coding agents (kept current)
├── docs/                       # This document + prod-deploy.md (launchd)
│
├── app/
│   ├── main.py                 # FastAPI app factory, lifespan (sweep, load models on MLX thread)
│   ├── config.py               # Pydantic BaseSettings — single source of truth
│   ├── dependencies.py         # Singletons + Annotated deps (SettingsDep, GateDep, ...)
│   │
│   ├── core/
│   │   ├── logging.py          # Loguru: text/JSON, rotating file, stdlib interception
│   │   ├── exceptions.py       # Domain errors, each with status_code + default_message
│   │   ├── error_handler.py    # Global handlers → {error, message, request_id}
│   │   ├── middleware.py       # RequestContextMiddleware (pure ASGI): request id + timing
│   │   ├── security.py         # require_api_key dependency (bearer / X-API-Key)
│   │   ├── gate.py             # JobGate — one GPU job at a time, bounded queue, 503 + Retry-After
│   │   └── mlx_worker.py       # MlxWorker — the single thread that loads models and runs inference
│   │
│   ├── api/
│   │   ├── health.py           # /health (503 until ready), /health/model — outside the API key
│   │   └── v1/
│   │       ├── router.py       # /api/v1 aggregate, API-key dependency
│   │       ├── transcribe.py   # /transcribe, /transcribe/url, /audio/transcriptions (one pipeline)
│   │       ├── embeddings.py   # /embeddings (OpenAI shape, base64, dimensions)
│   │       └── models.py       # /models
│   │
│   ├── services/
│   │   ├── transcriber.py      # mlx-whisper wrapper: real warm-up, TranscribeOptions, numpy input
│   │   ├── embedder.py         # mlx-embeddings wrapper: token limits, batching, EmbedResult
│   │   └── media.py            # ffmpeg → WAV → float32 array (single decode), DecodedAudio
│   │
│   ├── schemas/
│   │   ├── transcription.py    # TranscribeParams / UrlRequest / Response, Segment, Word, OutputFormat
│   │   └── embedding.py        # EmbeddingRequest / Response, ModelList
│   │
│   └── utils/
│       ├── file_handler.py     # Streaming upload save, magic bytes, SSRF-guarded download, temp sweep
│       └── formats.py          # SRT / VTT rendering
│
└── tests/
    ├── conftest.py             # make_client(**settings) with mocked services + fresh gate
    ├── unit/                   # config, gate, worker, media, transcriber, embedder, file_handler,
    │                           # formats, logging, middleware, dependencies
    └── integration/            # transcribe, embeddings, health, auth, errors, busy, lifespan
```

---

## 4. Configuration

### 4.1 Pydantic BaseSettings (`config.py`)

All configuration is driven by environment variables through Pydantic's `BaseSettings`. This provides type safety, validation, and automatic `.env` file loading.

```
Settings
  ├── App
  │   ├── app_name: str          = "whisperapy-mac"
  │   ├── debug: bool            = False
  │   └── version (property)     # from package metadata, not configurable
  │
  ├── Security
  │   ├── api_key: str | None    = None   # bearer / X-API-Key on /api/v1 when set
  │   ├── cors_origins: list     = []     # empty disables CORS
  │   └── allow_private_urls     = False  # SSRF guard on /transcribe/url
  │
  ├── Logging
  │   ├── log_format             = "text" | "json"
  │   └── log_file / log_rotation / log_retention
  │
  ├── Models
  │   ├── model_repo             = "mlx-community/whisper-large-v3-turbo"
  │   ├── embedding_model_repo   = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
  │   ├── embedding_max_tokens   = 8192   # 422 above this unless embedding_truncate
  │   ├── embedding_truncate     = False
  │   ├── embedding_max_batch    = 2048   # strings per request
  │   └── embedding_batch_size   = 32     # strings per GPU forward pass
  │
  ├── Concurrency
  │   ├── max_queued_jobs: int           = 1     # requests allowed to wait for the GPU
  │   ├── queue_wait_seconds: float      = 15.0  # max wait before 503
  │   ├── transcribe_speed_factor: float = 8.0   # x real-time, drives Retry-After
  │   └── embed_tokens_per_second        = 2000  # drives Retry-After for embed jobs
  │
  ├── File Handling
  │   ├── max_file_size_mb: int  = 1500
  │   ├── temp_dir: Path         = "/tmp/whisperapy"
  │   ├── temp_max_age_hours     = 6      # startup sweep of crash leftovers
  │   ├── ffmpeg_timeout_seconds = 120  (+ ffmpeg_timeout_seconds_per_mb = 0.5)
  │   ├── download_connect_timeout / download_read_timeout = 15 / 120
  │   └── allowed_extensions     = [mp4, mov, mkv, avi, webm,
  │                                  mp3, wav, m4a, ogg, flac, aac]
  │
  └── model_config               # .env, extra="ignore" (retired keys don't break boot)
```

### 4.2 `.env.example`

The committed `.env.example` lists every setting above with its default, so an
empty `.env` and the example file behave identically. `HOST` and `PORT` are not
settings; `make dev` reads them from the shell environment.

### 4.3 `pyproject.toml` Structure

```toml
[project]
name = "whisperapy-mac"
version = "1.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi", "uvicorn", "mlx-whisper", "mlx-embeddings", "numpy",
  "python-multipart", "pydantic", "pydantic-settings", "loguru", "httpx",
]   # each with a lower bound; uv.lock pins exact versions

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "pytest-cov", "ruff", "pyright"]

[build-system]
requires = ["hatchling"]

[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "ASYNC", "RUF"]

[tool.pyright]
include = ["app"]
typeCheckingMode = "standard"

[tool.pytest.ini_options]
asyncio_mode = "auto"
addopts = "--cov=app --cov-report=term-missing --cov-fail-under=90"

[tool.ruff.isort]
known-first-party = ["app"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## 5. API Endpoints

### 5.1 Health Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server alive check + model status + uptime + GPU busy state |
| `GET` | `/health/model` | Detailed model readiness and metadata |

**`GET /health` — Response**

```json
{
  "status": "ok",
  "version": "1.1.0",
  "model": "whisper-large-v3-turbo",
  "model_loaded": true,
  "embedding_model": "Qwen3-Embedding-4B-4bit-DWQ",
  "embedding_model_loaded": true,
  "uptime_seconds": 3600,
  "busy": true,
  "active_jobs": 1,
  "queued_jobs": 0,
  "max_concurrent_jobs": 1,
  "max_queued_jobs": 1,
  "estimated_wait_seconds": 149
}
```

`/health` always answers immediately, even mid-transcription, because no model
work runs on the event loop (see §6.4). It returns **503** with
`"status": "starting"` until both models are loaded, so monitors can tell
booting from broken. `estimated_wait_seconds` is `null` when idle. Health
routes are mounted outside the v1 router and are never behind the API key.

### 5.2 Transcription Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/transcribe` | Upload file (multipart), wait, receive transcript |
| `POST` | `/api/v1/transcribe/url` | JSON body with a public URL; busy check runs *before* download |
| `POST` | `/api/v1/audio/transcriptions` | OpenAI-compatible multipart shape (`file`, `model`, `language`, `prompt`, `response_format`, `temperature`, `timestamp_granularities[]`) |
| `GET` | `/api/v1/models` | OpenAI-compatible model list |

All three transcription routes share one pipeline (`_transcribe_source` in
`api/v1/transcribe.py`); they differ only in how the input file is obtained and
how form fields map onto `TranscribeParams`. There are no async job routes:
requests are synchronous and `JobStatus` was removed.

### 5.3 Embedding Endpoint

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/embeddings` | OpenAI-compatible — embed a string or list of strings |

**Request** (`application/json`):

```json
{ "input": ["hello world", "goodbye"], "encoding_format": "float", "dimensions": 1024, "prompt": null }
```

`input` may hold up to `EMBEDDING_MAX_BATCH` strings; any string over
`EMBEDDING_MAX_TOKENS` tokens is a **422** (`InputTooLongError`) unless
`EMBEDDING_TRUNCATE=true`, in which case it is cut and counted in the response's
`truncated` field. `encoding_format=base64` returns little-endian float32 as
OpenAI does; `dimensions` applies Matryoshka truncation with re-normalisation.

**`EmbeddingResponse`** — OpenAI-compatible envelope:

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.02] }
  ],
  "model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 },
  "truncated": 0
}
```

The model runs last-token pooling and returns L2-normalized vectors. An optional
`prompt` field prepends a Qwen3 instruction to each input (useful for query-side
retrieval embeddings) while staying OpenAI-compatible when omitted. `usage`
counts the tokens actually embedded (after any truncation). Inputs are sent to
the GPU in chunks of `EMBEDDING_BATCH_SIZE`.

### 5.4 Transcription Request Schema

`TranscribeParams` (form fields on the upload route, JSON fields on the URL route):

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` / `url` | UploadFile / HttpUrl | Yes | — | Media file, or a public http(s) URL |
| `language` | str | No | `auto` | Language code (e.g. `en`, `zh`) or `auto` |
| `include_segments` | bool | No | `false` | Include segment timings in JSON output |
| `word_timestamps` | bool | No | `false` | Populate `segments[].words` |
| `initial_prompt` | str | No | — | Vocabulary / context hint for the first window |
| `temperature` | float 0–1 | No | Whisper fallback ladder | Fix a single decoding temperature |
| `condition_on_previous_text` | bool | No | `true` | Feed prior text as context |
| `output_format` | enum | No | `json` | `json` \| `verbose_json` \| `text` \| `srt` \| `vtt` |

### 5.5 Response Schemas

**`TranscribeResponse`**

```json
{
  "job_id": "abc-123",
  "language_detected": "en",
  "duration_seconds": 124.5,
  "processing_time_seconds": 8.2,
  "text": "Full transcript here...",
  "segments": [
    { "start": 0.0, "end": 3.2, "text": "Hello world",
      "words": [ { "start": 0.0, "end": 0.4, "word": " Hello", "probability": 0.98 } ] },
    { "start": 3.2, "end": 6.1, "text": "How are you?", "words": [] }
  ]
}
```

`duration_seconds` is exact (length of the decoded audio), not the end of the
last segment. `words` is populated only when `word_timestamps=true`. With
`output_format=text|srt|vtt` the body is rendered by `utils/formats.py` and the
`Content-Type` is `text/plain`, `application/x-subrip`, or `text/vtt`.

### 5.6 Error Response Shape

All errors return a consistent JSON envelope — stack traces are never exposed to clients.

```json
{
  "error": "UnsupportedFormatError",
  "message": "File type .xyz is not supported",
  "request_id": "abc-123"
}
```

**Busy (503)** — returned when the GPU is occupied and the request cannot be
queued (see §6.4). Adds a `Retry-After` header and `retry_after_seconds`:

```json
{
  "error": "ServiceBusyError",
  "message": "Server is busy processing another request. Please retry later.",
  "request_id": "abc-123",
  "retry_after_seconds": 149
}
```

---

## 6. Processing Pipeline

### 6.1 Request Flow

```
Client uploads file (multipart/form-data)
          ↓
Middleware: inject Request ID (UUID), start timer
          ↓
security.py: require_api_key (401) when API_KEY is set
          ↓
gate.py: JobGate.reserve() — 503 immediately if the pipeline is full or the
         running job will outlast QUEUE_WAIT_SECONDS (for /transcribe/url this
         happens *before* the download; an upload has already been received)
          ↓
file_handler.py: extension + 16-byte magic peek + declared size, then stream
                 the upload to TEMP_DIR in 1 MB chunks under the size cap
                 (URL route: scheme + public-host check, streamed download)
          ↓
media.py: ffmpeg → 16kHz mono WAV, then load WAV → float32 numpy array
          (asyncio.to_thread; one decode, exact duration)
          ↓
gate.py: JobGate.run(estimated_seconds = duration / TRANSCRIBE_SPEED_FACTOR)
         — wait ≤ QUEUE_WAIT_SECONDS for the GPU slot, else 503
          ↓
mlx_worker.py: transcriber.transcribe(samples, options) on the MLX thread
          ↓
file_handler.py: cleanup tmp files (always, in a finally)
          ↓
render(): json / verbose_json / text / srt / vtt
          ↓
Return response + X-Request-ID + X-Processing-Time headers
```

### 6.2 Model Lifecycle (Lifespan)

The FastAPI lifespan context manager handles model loading and cleanup.

| Event | Action |
|---|---|
| Startup | Create `TEMP_DIR`, sweep files older than `TEMP_MAX_AGE_HOURS`, create the `MlxWorker` thread, load mlx-whisper (warm-up on one second of silence) and the embedding model **on that thread**. A load failure aborts startup. |
| Ready | Models are singletons — all requests share one loaded instance, no per-request cost |
| Shutdown | Shut down the MLX worker thread |

> **Performance Note:** Without lifespan management, every first request pays a ~3 second model load cost. With singleton loading at startup, all requests hit the already-warm model.

### 6.4 Threading & Concurrency

**MLX thread affinity.** MLX (0.32+) keeps GPU streams per thread. A model that
was loaded or warmed up on one thread cannot be evaluated from another; the
process aborts with `There is no Stream(gpu, N) in current thread`. Therefore
`MlxWorker` (`core/mlx_worker.py`) owns a single `ThreadPoolExecutor(max_workers=1)`
and *every* MLX call — both `load()`s at startup and every `transcribe()` /
`embed()` — runs on it via `await worker.run(fn, ...)`. Nothing MLX-related ever
runs on the event loop or in `asyncio.to_thread`. ffmpeg, WAV loading, and the
CPU tokenizer (`embedder.count_tokens`) are not MLX and use `asyncio.to_thread`,
so token counting never queues behind a job that holds the GPU.

**One GPU job at a time.** `JobGate` (`core/gate.py`) is the admission layer:

| Situation | Outcome |
|---|---|
| GPU idle | request runs immediately |
| GPU busy, queue slot free, running job expected to finish within `QUEUE_WAIT_SECONDS` | request waits, then runs |
| GPU busy, running job expected to take longer than `QUEUE_WAIT_SECONDS` | **503 immediately** (before any download) |
| GPU busy and `MAX_QUEUED_JOBS` already waiting | **503 immediately** |
| Queued request still waiting after `QUEUE_WAIT_SECONDS` | **503** |

Every job carries an `estimated_seconds`: `audio_seconds / TRANSCRIBE_SPEED_FACTOR`
for transcription (exact length of the decoded audio) and
`tokens / EMBED_TOKENS_PER_SECOND` for embeddings. The remaining time is that
estimate plus 2 s overhead minus elapsed. Transcription and embedding requests
share the same gate. Every 503 carries a
`Retry-After` header (minimum 5 s). A client that disconnects does not cancel
the running job — mlx-whisper cannot be interrupted.

### 6.3 Supported Formats

| Category | Formats |
|---|---|
| Video | `mp4`, `mov`, `avi`, `mkv`, `webm` |
| Audio | `mp3`, `wav`, `m4a`, `ogg`, `flac`, `aac` |

ffmpeg handles all conversion — any input is extracted to a 16kHz mono WAV before transcription.

---

## 7. Core Layer

### 7.1 Logging (`core/logging.py`)

Loguru is configured for structured logging with per-request context.

- Request ID attached to every log line for end-to-end tracing (bound by the middleware via `logger.contextualize`)
- Stdout sink always; optional rotating file sink via `LOG_FILE` / `LOG_ROTATION` / `LOG_RETENTION`
- `LOG_FORMAT=json` switches both sinks to loguru's serialized JSON
- uvicorn and httpx stdlib loggers are intercepted so every line shares one format
- Log levels: `DEBUG` when `DEBUG=true`, else `INFO`
- Every transcription logs: source, language, duration, processing time

### 7.2 Exception Hierarchy (`core/exceptions.py`)

```
WhisperapyError (base; each subclass declares status_code + default_message)
  ├── AuthenticationError      # missing / wrong API key               → 401
  ├── FileTooLargeError        # file exceeds MAX_FILE_SIZE_MB         → 413
  ├── UnsupportedFormatError   # extension or magic bytes not allowed  → 415
  ├── FileValidationError      # malformed upload                      → 422
  ├── DownloadError            # file download from URL failed         → 422
  ├── ForbiddenUrlError        # non-http scheme or non-public host    → 422
  ├── InvalidRequestError      # violates a configured limit (batch)   → 422
  ├── InputTooLongError        # embedding input > EMBEDDING_MAX_TOKENS→ 422
  ├── AudioExtractionError     # ffmpeg failed                         → 500
  ├── TranscriptionError       # mlx-whisper failed                    → 500
  ├── EmbeddingError           # mlx-embeddings failed                 → 500
  ├── ModelNotReadyError       # model not yet loaded at startup       → 503
  └── ServiceBusyError         # GPU busy; carries retry_after         → 503 + Retry-After
```

### 7.3 Middleware (`core/middleware.py`)

| Middleware | Function |
|---|---|
| `RequestContextMiddleware` | Pure ASGI (no `BaseHTTPMiddleware`). Honours an incoming `X-Request-ID` or generates a UUID, adds `X-Request-ID` and `X-Processing-Time` to the response, binds the id into loguru's context |
| `CORSMiddleware` | Added only when `CORS_ORIGINS` is non-empty — explicit origins, never a wildcard |

The unhandled-exception handler runs in Starlette's outermost
`ServerErrorMiddleware`, outside ours, so it sets `X-Request-ID` itself.

### 7.4 Security & Validation

| Concern | Approach |
|---|---|
| Unauthenticated GPU use | Optional `API_KEY`; `require_api_key` dependency on the v1 router (bearer or `X-API-Key`, constant-time compare). Health stays open |
| SSRF via `/transcribe/url` | `HttpUrl` typing, http/https only, host resolved and refused if loopback / private / link-local / reserved, re-checked on every redirect via an httpx response hook. `ALLOW_PRIVATE_URLS` opts out |
| File type spoofing | 16-byte magic peek; RIFF containers check the form type so `.avi` renamed `.wav` fails |
| Memory exhaustion | Uploads stream to disk in 1 MB chunks under the size cap; nothing holds a whole file in RAM |
| Path traversal | Sanitize filename, `tempfile.mkstemp` in `TEMP_DIR` |
| File size | `UploadFile.size` pre-check, then a running byte count while streaming |
| Stack trace leaks | Global error handler catches all exceptions, returns clean JSON |
| CORS | Off unless `CORS_ORIGINS` lists explicit origins |

---

## 8. Services

### 8.1 `TranscriberService` (`services/transcriber.py`)

- Singleton pattern — one model instance for the lifetime of the process
- Loaded during FastAPI lifespan startup, injected via `dependencies.py`
- Wraps mlx-whisper with a consistent input/output interface
- Takes decoded audio as a numpy array (never a path, which would make mlx-whisper decode again)
- `TranscribeOptions` carries language, word timestamps, initial prompt, temperature, condition_on_previous_text
- Maps raw mlx-whisper output to `TranscribeResponse` schema

```
TranscriberService
  ├── load()            # startup: transcribes 1 s of zeros → downloads, caches, compiles kernels;
  │                     # exceptions propagate so a broken model aborts startup
  ├── transcribe(
  │     audio: np.ndarray,
  │     duration_seconds: float | None,
  │     options: TranscribeOptions | None
  │   ) -> TranscribeResponse
  └── is_ready()        # returns bool for health endpoint
```

### 8.2 `EmbedderService` (`services/embedder.py`)

- Singleton pattern — one embedding model instance loaded at startup, mirroring `TranscriberService`
- Wraps `mlx-embeddings` (`load` / `generate`) with a consistent interface
- Imports `mlx_embeddings` inside methods so non-Apple / CI machines can import the module
- Returns L2-normalized vectors (last-token pooling) plus the token count actually embedded
- Enforces `EMBEDDING_MAX_TOKENS` (422, or truncate + flag), passes `max_length` to `generate`, chunks by `EMBEDDING_BATCH_SIZE`, applies Matryoshka `dimensions`

```
EmbedderService
  ├── load()                       # called at startup, loads model + tokenizer
  ├── count_tokens(texts) -> list[int]   # CPU tokenizer; safe off the MLX thread
  ├── check_lengths(counts) -> int       # raises InputTooLongError or returns truncated count
  ├── embed(
  │     texts: list[str],
  │     prompt: str | None,
  │     dimensions: int | None
  │   ) -> EmbedResult(vectors, prompt_tokens, truncated)
  └── is_ready()
```

### 8.2a `JobGate` (`core/gate.py`) and `MlxWorker` (`core/mlx_worker.py`)

- `JobGate.reserve(kind)` — async context manager claimed for the whole request; fails fast with `ServiceBusyError`
- `JobGate.run(job, estimated_seconds)` — async context manager holding the single GPU slot for the model call
- `JobGate.estimate_transcribe(audio_seconds)` / `estimate_embed(tokens)` — turn work size into seconds for the estimate
- `JobGate.snapshot()` — the busy fields merged into `/health`
- `MlxWorker.run(fn, *args, **kwargs)` — awaitable; executes `fn` on the one MLX thread
- `MlxWorker.run_sync(fn, ...)` — blocking variant for non-async callers
- See §6.4 for why both exist.

### 8.3 `MediaService` (`services/media.py`)

- Runs the `ffmpeg` CLI via `subprocess` (`-nostdin`, timeout = `FFMPEG_TIMEOUT_SECONDS` + per-MB allowance)
- Output: 16kHz mono 16-bit WAV, then loaded with the stdlib `wave` module into a float32 numpy array
- One decode per request: the array goes straight to mlx-whisper, which would otherwise spawn ffmpeg again
- Raises `AudioExtractionError` on failure, timeout, or missing ffmpeg with a clean message

```
MediaService(settings)
  ├── extract_audio(input_path, output_path) -> Path
  └── extract_and_load(input_path, output_path) -> DecodedAudio(samples, sample_rate)
load_wav(path) -> DecodedAudio          # .duration_seconds is exact
```

### 8.4 `FileHandler` (`utils/file_handler.py`)

| Function | Responsibility |
|---|---|
| `validate_upload()` | Extension whitelist, declared-size check, 16-byte magic peek (`validate_magic_bytes`) |
| `save_temp_file()` | Stream the upload to `TEMP_DIR` in 1 MB chunks under the size cap; unique name via `mkstemp` |
| `download_file_from_url()` | `check_url_allowed` (scheme + public host, re-checked on redirects), streamed download under the size cap |
| `sweep_temp_dir()` | Startup removal of files older than `TEMP_MAX_AGE_HOURS` |
| `cleanup_temp()` | Remove temp files after transcription completes or fails |
| `sanitize_filename()` | Strip path separators, normalize characters |

---

## 9. Testing Strategy

### 9.1 Test Structure

| Type | File | What it Tests |
|---|---|---|
| Unit | `test_config.py` | Defaults, env parsing, retired keys ignored, version from metadata, timeout scaling |
| Unit | `test_file_handler.py` | Magic bytes per format, streamed save + size cap, temp sweep, SSRF guard incl. redirects, downloads |
| Unit | `test_transcriber.py` | Real warm-up, load failure propagation, options mapping, word timestamps |
| Unit | `test_embedder.py` | Token limit / truncation policy, batching, `max_length`, dimensions, usage |
| Unit | `test_gate.py` | Admission: queueing, fail-fast, timeouts, transcribe + embed estimates |
| Unit | `test_mlx_worker.py` | All calls land on the one MLX thread, off the event loop |
| Unit | `test_media.py` | `load_wav`, ffmpeg command + scaled timeout, failure modes, single decode |
| Unit | `test_formats.py` | SRT / VTT rendering |
| Unit | `test_logging.py` | Text/JSON sinks, rotating file, stdlib interception |
| Unit | `test_middleware.py`, `test_dependencies.py` | ASGI passthrough + headers; lazy singletons |
| Integration | `test_health.py` | Ready vs `starting` (503), model endpoint, CORS wiring, health open without key |
| Integration | `test_transcribe.py` | Upload / URL / OpenAI routes, every output format, options plumbing, temp cleanup |
| Integration | `test_embeddings.py` | OpenAI shape, base64, dimensions, batch + length limits, `/models` |
| Integration | `test_auth.py` | 401 without / with wrong key; bearer and `X-API-Key` accepted |
| Integration | `test_errors.py` | Every domain error → status + envelope, `Retry-After`, 500 without details, request-id echo |
| Integration | `test_busy.py` | Concurrent requests: health stays responsive, 503 + Retry-After, embed estimates |
| Integration | `test_lifespan.py` | Models load on the MLX thread; load failure aborts; temp sweep |

200 tests, ~2 s, 99% line coverage; the gate in `pyproject.toml` is 90%.

### 9.2 Tools

- `pytest` + `pytest-cov` — test runner with a coverage gate
- `pytest-asyncio` (`asyncio_mode = "auto"`) — async test support
- `httpx` `AsyncClient` over `ASGITransport(raise_app_exceptions=False)` — so the app's own 500 handler is tested
- `conftest.py` — `make_client(**settings_overrides)` builds an app with mocked transcriber / media / embedder, a fresh `JobGate`, and `dependency_overrides[get_settings]`; MLX modules are mocked via `sys.modules`

---

## 10. Makefile Commands

| Command | Action |
|---|---|
| `make dev` | `uv run uvicorn app.main:app --reload --host $(HOST) --port $(PORT)` |
| `make test` | `uv run pytest` (coverage gate 90%) |
| `make lint` | `uv run ruff check .` + `uv run ruff format --check .` |
| `make format` | `uv run ruff check --fix .` + `uv run ruff format .` |
| `make typecheck` | `uv run pyright` |
| `make check` | lint + typecheck + test — also what CI runs |
| `make clean` | Delete `$(TEMP_DIR)` contents and tool caches |
| `make install` | `uv sync` — install all dependencies |
| `make install-dev` | `uv sync --extra dev` — include dev dependencies |

---

## 11. Best Practices Checklist

### Must Have (v1)

| Item | Location | Priority |
|---|---|---|
| Lifespan model loading (singleton) | `main.py` | 🔴 Critical |
| All MLX calls on one dedicated thread | `core/mlx_worker.py` | 🔴 Critical |
| GPU admission gate — fail fast with 503 + Retry-After | `core/gate.py` | 🔴 Critical |
| Global exception handlers | `core/error_handler.py` | 🔴 Critical |
| Magic byte file validation | `utils/file_handler.py` | 🔴 Critical |
| Pydantic BaseSettings for all config | `config.py` | 🔴 Critical |
| Structured logging with Loguru | `core/logging.py` | 🟡 Important |
| Request ID middleware | `core/middleware.py` | 🟡 Important |
| Consistent error response shape | `core/exceptions.py` | 🟡 Important |
| Makefile developer shortcuts | `Makefile` | 🟡 Important |
| Ruff + pyright + coverage gate in `pyproject.toml`, CI | `pyproject.toml`, `.github/workflows/ci.yml` | 🟡 Important |
| Streaming uploads, single audio decode, real warm-up | `utils/file_handler.py`, `services/media.py`, `services/transcriber.py` | 🔴 Critical |
| API key + SSRF guard when exposed beyond localhost | `core/security.py`, `utils/file_handler.py` | 🔴 Critical |

### Nice to Have (v2)

| Item | Notes |
|---|---|
| Rate limiting | Not done. `API_KEY` auth landed in v1.1; add `slowapi` if a shared deployment needs per-client limits |
| pytest test suite | **Done (v1.1)** — 200 tests, 90% coverage gate, CI on macOS |
| Speaker diarization | Identify different speakers in audio |
| WebSocket streaming | Real-time transcription as audio is processed |
| Batch processing endpoint | Accept multiple files in one request |
| Model selection per request | Not done. `model` is accepted on the OpenAI routes but the server model is used |

---

## 12. Implementation Order

Build in this order to ensure each layer has its dependencies in place:

1. `pyproject.toml` — foundation, dependencies, ruff / pyright / pytest config
2. `config.py` — Pydantic BaseSettings, all other modules depend on this
3. `core/` — logging, exceptions, error handlers, middleware
4. `main.py` — FastAPI app init, lifespan, router registration
5. `services/` — transcriber singleton + media ffmpeg wrapper
6. `schemas/` — Pydantic request/response models
7. `api/v1/` — endpoints wired to services
8. `utils/` — file handler, validation, cleanup
9. `tests/` — unit tests for services, integration tests for endpoints

---

*whisperapy-mac — Design Document v1.1*
