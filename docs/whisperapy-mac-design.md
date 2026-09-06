# whisperapy-mac

> **Mac Native · FastAPI · mlx-whisper · Apple Silicon M4**
> Project Design Document — Version 1.0

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
| Audio Extraction | `ffmpeg` + `ffmpeg-python` | Handles any video/audio format (note: largely unmaintained, fallback to subprocess if needed) |
| File Uploads | `python-multipart` | Required by FastAPI for multipart/form-data |
| Settings | `Pydantic BaseSettings` | Type-safe env config |
| Validation | `Pydantic v2` | Request/response schemas |
| Logging | `Loguru` | Structured, easy async logging |
| Formatting | `Black` | Opinionated, consistent style |
| Linting | `Ruff` | Fast linting + import sorting |
| Testing | `pytest` + `pytest-asyncio` + `httpx` | Async test client for FastAPI |
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
├── pyproject.toml              # uv + black + ruff + all dependencies
├── .python-version             # Pin Python 3.12
├── .env                        # Local env vars (gitignored)
├── .env.example                # Committed template — no values
├── .gitignore                  # tmp/, .env, __pycache__, .venv
├── Makefile                    # Developer shortcuts
├── README.md
│
├── app/
│   ├── main.py                 # FastAPI app init, lifespan
│   ├── config.py               # Pydantic BaseSettings — single source of truth
│   ├── dependencies.py         # Shared FastAPI deps (get_settings, get_transcriber, get_gate, ...)
│   │
│   ├── core/
│   │   ├── logging.py          # Loguru structured logging setup
│   │   ├── exceptions.py       # Custom exception classes
│   │   ├── error_handler.py    # Global FastAPI exception handlers
│   │   ├── middleware.py       # Request ID injection, timing headers
│   │   ├── gate.py             # JobGate — one GPU job at a time, bounded queue, 503 + Retry-After
│   │   └── mlx_worker.py       # MlxWorker — the single thread that loads models and runs inference
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py       # Aggregates all v1 routes
│   │       └── transcribe.py   # All transcription endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcriber.py      # mlx-whisper singleton wrapper
│   │   └── media.py            # ffmpeg audio extraction logic
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── transcription.py    # Pydantic request/response models
│   │
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py     # Upload validation, magic bytes, cleanup
│
├── tests/
│   ├── conftest.py             # Shared fixtures, mock model
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_file_handler.py
│   │   └── test_transcriber.py
│   └── integration/
│       ├── test_health.py
│       └── test_transcribe.py
│
└── tmp/                        # Temp processing files (gitignored)
```

---

## 4. Configuration

### 4.1 Pydantic BaseSettings (`config.py`)

All configuration is driven by environment variables through Pydantic's `BaseSettings`. This provides type safety, validation, and automatic `.env` file loading.

```
Settings
  ├── App
  │   ├── app_name: str          = "whisperapy-mac"
  │   ├── version: str           = "1.0.0"
  │   ├── debug: bool            = False
  │   ├── host: str              = "0.0.0.0"
  │   └── port: int              = 8000
  │
  ├── Model
  │   ├── model_repo: str           = "mlx-community/whisper-large-v3-turbo"
  │   ├── default_language: str     = "auto"
  │   └── embedding_model_repo: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
  │
  ├── Concurrency
  │   ├── max_queued_jobs: int           = 1     # requests allowed to wait for the GPU
  │   ├── queue_wait_seconds: float      = 15.0  # max wait before 503
  │   └── transcribe_speed_factor: float = 8.0   # x real-time, drives Retry-After
  │
  ├── File Handling
  │   ├── max_file_size_mb: int  = 500
  │   ├── temp_dir: Path         = "./tmp"
  │   └── allowed_extensions     = [mp4, mov, mkv, avi, webm,
  │                                  mp3, wav, m4a, ogg, flac, aac]
  │
  └── model_config               # reads from .env file automatically
```

### 4.2 `.env.example`

```env
APP_NAME=whisperapy-mac
VERSION=1.0.0
DEBUG=false
HOST=0.0.0.0
PORT=8000

MODEL_REPO=mlx-community/whisper-large-v3-turbo
DEFAULT_LANGUAGE=auto
EMBEDDING_MODEL_REPO=mlx-community/Qwen3-Embedding-4B-4bit-DWQ

MAX_FILE_SIZE_MB=500
TEMP_DIR=./tmp

MAX_QUEUED_JOBS=1
QUEUE_WAIT_SECONDS=15
TRANSCRIBE_SPEED_FACTOR=8
```

### 4.3 `pyproject.toml` Structure

```toml
[project]
name = "whisperapy-mac"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi",
  "uvicorn",
  "mlx-whisper",
  "mlx-embeddings",
  "ffmpeg-python",
  "python-multipart",
  "pydantic-settings",
  "loguru",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "httpx", "black", "ruff"]

[build-system]
requires = ["hatchling"]

[tool.black]
line-length = 88

[tool.ruff]
line-length = 88
select = ["E", "F", "I"]    # pycodestyle, pyflakes, isort

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
  "version": "1.0.0",
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
work runs on the event loop (see §6.4). `estimated_wait_seconds` is `null` when
idle or when the running job's length is unknown (embeddings).

### 5.2 Transcription Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/transcribe` | Sync — upload file, wait, receive transcript |
| `POST` | `/api/v1/transcribe/url` | Sync — pass a URL, download + transcribe |
| `POST` | `/api/v1/transcribe/jobs` | Async — upload file, receive `job_id` immediately |
| `GET` | `/api/v1/transcribe/jobs/{job_id}` | Poll job status and retrieve result |
| `DELETE` | `/api/v1/transcribe/jobs/{job_id}` | Cancel job and clean up temp files |

### 5.3 Embedding Endpoint

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/embeddings` | OpenAI-compatible — embed a string or list of strings |

**Request** (`application/json`):

```json
{ "input": ["hello world", "goodbye"] }
```

**`EmbeddingResponse`** — OpenAI-compatible envelope:

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.02] }
  ],
  "model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

The model runs last-token pooling and returns L2-normalized vectors. An optional
`prompt` field prepends a Qwen3 instruction to each input (useful for query-side
retrieval embeddings) while staying OpenAI-compatible when omitted.

### 5.4 Transcription Request Schema

All transcription requests use `multipart/form-data`:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | UploadFile | Yes | — | Video or audio file |
| `language` | str | No | `auto` | Language code (e.g. `en`, `zh`) |
| `word_timestamps` | bool | No | `false` | Include word-level timestamps |
| `output_format` | enum | No | `json` | `json` \| `text` \| `srt` \| `vtt` |

### 5.5 Response Schemas

**`TranscribeResponse`**

```json
{
  "job_id": "abc-123",
  "status": "completed",
  "language_detected": "en",
  "duration_seconds": 124.5,
  "processing_time_seconds": 8.2,
  "text": "Full transcript here...",
  "segments": [
    { "start": 0.0, "end": 3.2, "text": "Hello world" },
    { "start": 3.2, "end": 6.1, "text": "How are you?" }
  ]
}
```

**`JobStatusResponse`**

```json
{
  "job_id": "abc-123",
  "status": "processing",
  "progress": 0.65,
  "error": null,
  "result": null
}
```

Status values: `pending` | `processing` | `completed` | `failed`

> **Note:** Async job state is stored in-memory only. Jobs are lost on server restart. This is acceptable for v1 as a local service.

### 5.6 Error Response Shape

All errors return a consistent JSON envelope — stack traces are never exposed to clients.

```json
{
  "error": "UnsupportedFormat",
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
gate.py: JobGate.reserve() — 503 immediately if the pipeline is full or the
         running job will outlast QUEUE_WAIT_SECONDS (for /transcribe/url this
         happens *before* the download)
          ↓
file_handler.py: validate extension + magic bytes + file size
          ↓
media.py: ffmpeg extract audio → 16kHz mono WAV → tmp/   (asyncio.to_thread)
          ↓
gate.py: JobGate.run() — wait ≤ QUEUE_WAIT_SECONDS for the GPU slot, else 503
          ↓
mlx_worker.py: run transcriber.transcribe() on the dedicated MLX thread
          ↓
Schema mapping: raw output → TranscribeResponse
          ↓
file_handler.py: cleanup tmp files
          ↓
Return response + X-Request-ID + X-Processing-Time headers
```

### 6.2 Model Lifecycle (Lifespan)

The FastAPI lifespan context manager handles model loading and cleanup.

| Event | Action |
|---|---|
| Startup | Create `tmp/`, create the `MlxWorker` thread, load mlx-whisper and the embedding model **on that thread** |
| Ready | Models are singletons — all requests share one loaded instance, no per-request cost |
| Shutdown | Flush Loguru logs, shut down the MLX worker thread |

> **Performance Note:** Without lifespan management, every first request pays a ~3 second model load cost. With singleton loading at startup, all requests hit the already-warm model.

### 6.4 Threading & Concurrency

**MLX thread affinity.** MLX (0.32+) keeps GPU streams per thread. A model that
was loaded or warmed up on one thread cannot be evaluated from another; the
process aborts with `There is no Stream(gpu, N) in current thread`. Therefore
`MlxWorker` (`core/mlx_worker.py`) owns a single `ThreadPoolExecutor(max_workers=1)`
and *every* MLX call — both `load()`s at startup and every `transcribe()` /
`embed()` — runs on it via `await worker.run(fn, ...)`. Nothing MLX-related ever
runs on the event loop or in `asyncio.to_thread`. ffmpeg is a plain subprocess and
does use `asyncio.to_thread`.

**One GPU job at a time.** `JobGate` (`core/gate.py`) is the admission layer:

| Situation | Outcome |
|---|---|
| GPU idle | request runs immediately |
| GPU busy, queue slot free, running job expected to finish within `QUEUE_WAIT_SECONDS` | request waits, then runs |
| GPU busy, running job expected to take longer than `QUEUE_WAIT_SECONDS` | **503 immediately** (before any download) |
| GPU busy and `MAX_QUEUED_JOBS` already waiting | **503 immediately** |
| Queued request still waiting after `QUEUE_WAIT_SECONDS` | **503** |

The estimate is `wav_seconds / TRANSCRIBE_SPEED_FACTOR + 2s` minus elapsed
time, where `wav_seconds` comes from the extracted 16 kHz mono WAV's size.
Transcription and embedding requests share the same gate. Every 503 carries a
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

- Request ID attached to every log line for end-to-end tracing
- Stdout in development (human-readable), file sink in production
- Log levels: `DEBUG` locally, `INFO` in production — controlled via `.env`
- Every transcription logs: filename, size, language, model, processing time

### 7.2 Exception Hierarchy (`core/exceptions.py`)

```
WhisperapyError (base)
  ├── FileTooLargeError        # file exceeds MAX_FILE_SIZE_MB
  ├── UnsupportedFormatError   # extension or magic bytes not allowed
  ├── FileValidationError      # malformed upload
  ├── AudioExtractionError     # ffmpeg failed
  ├── TranscriptionError       # mlx-whisper failed
  ├── DownloadError            # file download from URL failed
  ├── EmbeddingError           # mlx-embeddings failed
  ├── ModelNotReadyError       # model not yet loaded at startup       → 503
  └── ServiceBusyError         # GPU busy; carries retry_after         → 503 + Retry-After
```

### 7.3 Middleware (`core/middleware.py`)

| Middleware | Function |
|---|---|
| `RequestIDMiddleware` | Injects UUID into each request, adds `X-Request-ID` to response |
| `TimingMiddleware` | Measures total request duration, adds `X-Processing-Time` to response |
| `CORSMiddleware` | Explicit allowed origins — no wildcard in production |

### 7.4 Security & Validation

| Concern | Approach |
|---|---|
| File type spoofing | Validate magic bytes (file header), not just extension |
| Path traversal | Sanitize filename with `pathlib` before saving to `tmp/` |
| File size | Reject before reading body — checked via `Content-Length` header |
| Stack trace leaks | Global error handler catches all exceptions, returns clean JSON |
| CORS | Explicit allowed origins configured via settings (default: `localhost` only) |

---

## 8. Services

### 8.1 `TranscriberService` (`services/transcriber.py`)

- Singleton pattern — one model instance for the lifetime of the process
- Loaded during FastAPI lifespan startup, injected via `dependencies.py`
- Wraps mlx-whisper with a consistent input/output interface
- Handles language detection vs explicit language parameter
- Maps raw mlx-whisper output to `TranscribeResponse` schema

```
TranscriberService
  ├── load()            # called at startup, loads model from HuggingFace cache
  ├── transcribe(
  │     audio_path,
  │     language,
  │     word_timestamps
  │   ) -> TranscribeResponse
  └── is_ready()        # returns bool for health endpoint
```

### 8.2 `EmbedderService` (`services/embedder.py`)

- Singleton pattern — one embedding model instance loaded at startup, mirroring `TranscriberService`
- Wraps `mlx-embeddings` (`load` / `generate`) with a consistent interface
- Imports `mlx_embeddings` inside methods so non-Apple / CI machines can import the module
- Returns L2-normalized vectors (last-token pooling) plus a best-effort token count for `usage`

```
EmbedderService
  ├── load()            # called at startup, loads model + tokenizer
  ├── embed(
  │     texts: list[str],
  │     prompt: str | None
  │   ) -> tuple[list[list[float]], int]   # (vectors, prompt_tokens)
  └── is_ready()        # returns bool for health endpoint
```

### 8.2a `JobGate` (`core/gate.py`) and `MlxWorker` (`core/mlx_worker.py`)

- `JobGate.reserve(kind)` — async context manager claimed for the whole request; fails fast with `ServiceBusyError`
- `JobGate.run(job, audio_seconds)` — async context manager holding the single GPU slot for the model call
- `JobGate.snapshot()` — the busy fields merged into `/health`
- `MlxWorker.run(fn, *args, **kwargs)` — awaitable; executes `fn` on the one MLX thread
- `MlxWorker.run_sync(fn, ...)` — blocking variant for non-async callers
- See §6.4 for why both exist.

### 8.3 `MediaService` (`services/media.py`)

- Wraps `ffmpeg-python` to extract audio from any input format
- Output: 16kHz mono WAV (optimal for Whisper)
- Uses `tmp/` for intermediate files
- Raises `AudioExtractionError` on ffmpeg failure with a clean message

```
MediaService
  └── extract_audio(
        input_path: Path,
        output_path: Path
      ) -> Path          # returns path to extracted WAV
```

### 8.4 `FileHandler` (`utils/file_handler.py`)

| Function | Responsibility |
|---|---|
| `validate_upload()` | Check extension, magic bytes, and file size limit |
| `save_temp_file()` | Save upload to `tmp/` with sanitized filename |
| `cleanup_temp()` | Remove temp files after transcription completes or fails |
| `sanitize_filename()` | Strip path separators, normalize characters |

---

## 9. Testing Strategy

### 9.1 Test Structure

| Type | File | What it Tests |
|---|---|---|
| Unit | `test_config.py` | Settings load from env, defaults, validation |
| Unit | `test_file_handler.py` | Magic byte checks, size limits, path sanitization |
| Unit | `test_transcriber.py` | Transcriber with mocked mlx-whisper |
| Unit | `test_embedder.py` | Embedder with mocked mlx-embeddings |
| Unit | `test_gate.py` | Admission: queueing, fail-fast, timeouts, Retry-After estimates |
| Unit | `test_mlx_worker.py` | All calls land on the one MLX thread, off the event loop |
| Unit | `test_media.py` | WAV duration estimate from file size |
| Integration | `test_health.py` | Health endpoints return correct status and busy fields |
| Integration | `test_transcribe.py` | Full upload and URL transcription flows |
| Integration | `test_embeddings.py` | OpenAI-shaped embeddings responses |
| Integration | `test_busy.py` | Concurrent requests: health stays responsive, 503 + Retry-After, queued request succeeds |
| Integration | `test_lifespan.py` | Startup loads both models on the MLX worker thread |

### 9.2 Tools

- `pytest` — test runner
- `pytest-asyncio` (`asyncio_mode = "auto"`) — async test support
- `httpx` `AsyncClient` — async HTTP client for FastAPI test client
- `conftest.py` — shared fixtures: test client, mock model, sample audio files

---

## 10. Makefile Commands

| Command | Action |
|---|---|
| `make dev` | `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` |
| `make test` | `uv run pytest tests/ -v` |
| `make lint` | `uv run ruff check .` |
| `make format` | `uv run black .` |
| `make check` | Run lint + format + test in sequence |
| `make clean` | Delete `tmp/` contents and `__pycache__` directories |
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
| Black + Ruff in `pyproject.toml` | `pyproject.toml` | 🟡 Important |

### Nice to Have (v2)

| Item | Notes |
|---|---|
| Rate limiting | Not done. `API_KEY` auth landed in v1.1; add `slowapi` if a shared deployment needs per-client limits |
| pytest test suite | **Done (v1.1)** — 180 tests, 90% coverage gate, CI on macOS |
| Speaker diarization | Identify different speakers in audio |
| WebSocket streaming | Real-time transcription as audio is processed |
| Batch processing endpoint | Accept multiple files in one request |
| Model selection per request | Not done. `model` is accepted on the OpenAI routes but the server model is used |

---

## 12. Implementation Order

Build in this order to ensure each layer has its dependencies in place:

1. `pyproject.toml` — foundation, dependencies, Black/Ruff config
2. `config.py` — Pydantic BaseSettings, all other modules depend on this
3. `core/` — logging, exceptions, error handlers, middleware
4. `main.py` — FastAPI app init, lifespan, router registration
5. `services/` — transcriber singleton + media ffmpeg wrapper
6. `schemas/` — Pydantic request/response models
7. `api/v1/` — endpoints wired to services
8. `utils/` — file handler, validation, cleanup
9. `tests/` — unit tests for services, integration tests for endpoints

---

*whisperapy-mac — Design Document v1.0*
