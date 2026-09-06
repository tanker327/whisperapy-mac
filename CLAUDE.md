# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install-dev      # uv sync --extra dev
make dev              # uvicorn with hot reload; honours HOST / PORT env vars
make test             # pytest with coverage (fails under 90%)
make lint             # ruff check + ruff format --check
make format           # ruff --fix + ruff format
make typecheck        # pyright (standard mode, app/ only)
make check            # lint + typecheck + test — run before every commit
make clean            # temp files and tool caches

# Run a single test file or test
uv run pytest tests/unit/test_gate.py -v --no-cov
uv run pytest tests/unit/test_transcriber.py::test_transcribe_returns_response -v --no-cov
```

CI (`.github/workflows/ci.yml`) runs `make check` on `macos-14`. Black is gone; ruff formats.

## Architecture

Local REST transcription + embedding service: FastAPI + mlx-whisper + mlx-embeddings on Apple Silicon. No Docker — requires native Metal GPU access. Requires `ffmpeg` on the system.

**API** (all under `/api/v1`, behind the optional API key):
`POST /transcribe` (multipart), `POST /transcribe/url` (JSON), `POST /audio/transcriptions` (OpenAI shape), `POST /embeddings` (OpenAI shape), `GET /models`. Health lives outside the router and is never behind the key: `GET /health` (503 + `status: starting` until both models are loaded), `GET /health/model`.

**Request flow (transcription, any source):** Middleware (request id, timing) → API-key dependency → endpoint builds `TranscribeParams` → `JobGate.reserve` (fail fast 503 if busy) → obtain input (`validate_upload` peeks 16 bytes for magic + streams to disk via `save_temp_file`, or `download_file_from_url` with SSRF guard) → `MediaService.extract_and_load` in `asyncio.to_thread` (ffmpeg → 16 kHz WAV → numpy float32; **one decode**, exact duration) → `JobGate.run` with `gate.estimate_transcribe(duration)` → `TranscriberService.transcribe(samples, ...)` on the `MlxWorker` thread → `cleanup_temp` → `render()` (json / verbose_json / text / srt / vtt). All three transcription routes share `_transcribe_source` in `app/api/v1/transcribe.py`.

**Request flow (embeddings):** batch-size check (422) → `embedder.count_tokens` in `asyncio.to_thread` (CPU tokenizer, deliberately *not* on the MLX thread so it never queues behind the GPU) → `JobGate.reserve` + `run` with `gate.estimate_embed(tokens)` → `EmbedderService.embed` on the `MlxWorker` thread (enforces `embedding_max_tokens`, chunks by `embedding_batch_size`, optional Matryoshka `dimensions`) → OpenAI-shaped response, `base64` encoding done in the endpoint.

**Dependency injection:** Services are module-level singletons in `app/dependencies.py`, created by `init_services()` during lifespan. Endpoints use the `Annotated` aliases (`SettingsDep`, `TranscriberDep`, `MediaDep`, `EmbedderDep`, `GateDep`, `WorkerDep`) — never `= Depends(...)` defaults (ruff B008). `get_gate()` and `get_mlx_worker()` build lazily if `init_services()` never ran (tests).

**Error handling:** Every domain error extends `WhisperapyError` and declares its own `status_code` and `default_message` (`app/core/exceptions.py`); the handler in `app/core/error_handler.py` has no lookup table. Response shape is always `{error, message, request_id}`; `ServiceBusyError` adds `Retry-After` + `retry_after_seconds`. The unhandled-exception handler sets `X-Request-ID` itself because it runs in Starlette's outermost middleware, outside ours. Stack traces are never exposed.

**Model lifecycle:** Lifespan sweeps stale temp files, then loads both models on the MLX worker. `TranscriberService.load` transcribes one second of zeros — a real warm-up that also compiles Metal kernels — and **lets failures propagate** so a server that cannot load its model never reports healthy. `mlx_whisper` and `mlx_embeddings` are imported inside methods so tests run anywhere.

**MLX thread affinity (critical):** MLX keeps GPU streams per thread. A model loaded on one thread cannot be evaluated from another — the process aborts with `There is no Stream(gpu, N) in current thread`. Every MLX call (both `load()`s and all inference) goes through the single-thread `MlxWorker` (`await worker.run(fn, ...)`). Never call `transcriber.transcribe` / `embedder.embed` / `load` from the event loop or `asyncio.to_thread`. ffmpeg, WAV loading, and tokenizer `encode` are not MLX and use `asyncio.to_thread`.

**Concurrency / busy handling:** `JobGate` (`app/core/gate.py`) admits one GPU job at a time with a bounded queue (`MAX_QUEUED_JOBS`, `QUEUE_WAIT_SECONDS`). `reserve()` fails fast when the pipeline is full or the running job's `estimated_seconds` exceeds the wait budget; `run()` holds the slot. Estimates: audio seconds / `TRANSCRIBE_SPEED_FACTOR`, tokens / `EMBED_TOKENS_PER_SECOND`. `/health` exposes `busy`, `active_jobs`, `queued_jobs`, `estimated_wait_seconds`. Uploads are already received by FastAPI before `reserve()` runs; only the URL route rejects before transfer.

**Security:** `API_KEY` (bearer or `X-API-Key`, constant-time compare) via `require_api_key` on the v1 router. `check_url_allowed` resolves the host and refuses non-public addresses, re-checked on every redirect via an httpx response hook; `ALLOW_PRIVATE_URLS` disables it. CORS is off unless `CORS_ORIGINS` is set.

**Middleware / logging:** One pure-ASGI `RequestContextMiddleware` (no `BaseHTTPMiddleware`) sets/honours `X-Request-ID`, adds `X-Processing-Time`, and binds the id into loguru's context. `setup_logging` supports `LOG_FORMAT=json`, a rotating `LOG_FILE`, and intercepts uvicorn/httpx stdlib loggers.

**Temp files:** `tempfile.mkstemp` in `TEMP_DIR` for inputs; ffmpeg output is `{stem}_extracted.wav` next to it. Both are removed in a `finally`; leftovers older than `TEMP_MAX_AGE_HOURS` are swept at startup.

## Key Conventions

- **Config:** Pydantic `BaseSettings` in `app/config.py`, `extra="ignore"` so retired `.env` keys don't break boot. `version` is a read-only property from package metadata — bump it in `pyproject.toml` only. Use `_env_file=None` in tests.
- **Formatting / lint:** ruff (line length 88, py312) with `E F I B UP SIM ASYNC RUF`. `known-first-party = ["app"]`.
- **Types:** pyright standard mode must pass on `app/`. Lazily-imported MLX objects are typed `Any`.
- **Testing:** pytest-asyncio in auto mode. `tests/conftest.py` provides `make_client(**settings_overrides)` / `env` which build an app with mocked `transcriber`, `media`, `embedder`, a fresh `JobGate`, `app.dependency_overrides[get_settings]`, and `raise_app_exceptions=False`. Mock MLX modules with `patch.dict(sys.modules, {"mlx_whisper": mock})`. Busy-path tests block the mocked transcriber on a `threading.Event`. Coverage gate is 90%.
- **File validation:** upload route: extension whitelist + 16-byte magic peek (RIFF checks the form type, so `.avi` renamed `.wav` fails) + streamed size cap. URL route: scheme + public-host check + size cap; ffmpeg validates the content.
