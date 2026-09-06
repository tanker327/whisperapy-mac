# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install-dev      # Install all deps including dev (uv sync --extra dev)
make dev              # Start dev server with hot reload (uvicorn, port 8000)
make test             # Run full test suite (pytest -v)
make lint             # Lint with ruff
make format           # Format with black
make check            # lint + format + test in sequence
make clean            # Remove tmp/ contents and __pycache__

# Run a single test file or test
uv run pytest tests/unit/test_config.py -v
uv run pytest tests/unit/test_transcriber.py::test_transcribe_returns_response -v
```

## Architecture

Local REST transcription service: FastAPI + mlx-whisper on Apple Silicon. No Docker — requires native Metal GPU access. Requires `ffmpeg` installed on the system.

**API:** `POST /api/v1/transcribe` (sync, multipart upload), `POST /api/v1/transcribe/url` (sync, JSON body with URL), `POST /api/v1/embeddings` (OpenAI-compatible, JSON body with `input`), `GET /health`, `GET /health/model`. Default transcription model: `mlx-community/whisper-large-v3-turbo`. Default embedding model: `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`.

**Request flow (upload):** Client → Middleware (request ID, timing) → Endpoint → `JobGate.reserve` (fail fast with 503 if busy) → `file_handler.validate_upload` (extension + magic bytes + size) → `save_temp_file` → `MediaService.extract_audio` (ffmpeg → 16kHz WAV, in `asyncio.to_thread`) → `JobGate.run` (wait for the GPU slot, bounded) → `TranscriberService.transcribe` on the `MlxWorker` thread → `cleanup_temp` → Response

**Request flow (URL):** Client → Middleware → Endpoint → `JobGate.reserve` (rejects *before* downloading when the running job is long) → `file_handler.download_file_from_url` (httpx streaming + size limit, no format validation) → `MediaService.extract_audio` → `JobGate.run` → `TranscriberService.transcribe` on the `MlxWorker` thread → `cleanup_temp` → Response

**Request flow (embeddings):** Client → Middleware → Endpoint (`app/api/v1/embeddings.py`) → normalize `input` (str or list) to a list → `JobGate.reserve` + `JobGate.run` (shares the single GPU slot with transcription) → `EmbedderService.embed` (mlx-embeddings / Qwen3) on the `MlxWorker` thread → OpenAI-shaped `{object, data:[{index, embedding}], model, usage}` Response

**Dependency injection:** Services are module-level singletons in `app/dependencies.py`. `init_services()` is called once during FastAPI lifespan startup. Endpoints inject via `Depends(get_transcriber)`, `Depends(get_media_service)`, `Depends(get_embedder)`, `Depends(get_gate)`, `Depends(get_mlx_worker)`, `Depends(get_settings)`. Settings use `@lru_cache`. `get_gate()` and `get_mlx_worker()` build lazily if `init_services()` never ran (tests).

**Error handling:** All domain errors extend `WhisperapyError` (in `app/core/exceptions.py`). Global handlers in `app/core/error_handler.py` map each subclass to an HTTP status code and return consistent `{error, message, request_id}` JSON. `ServiceBusyError` → 503 and additionally sets a `Retry-After` header plus `retry_after_seconds` in the body. Stack traces are never exposed.

**Model lifecycle:** Two models load once at startup via the lifespan context manager in `app/main.py` — the mlx-whisper transcription model (`TranscriberService`) and the mlx-embeddings Qwen3 model (`EmbedderService`). All requests share these singletons — no per-request load cost. Both `mlx_whisper` and `mlx_embeddings` are imported inside methods (not at top-level) to avoid import errors on non-Apple-Silicon machines and in tests.

**MLX thread affinity (critical):** MLX keeps GPU streams per thread. A model loaded or warmed up on one thread cannot be evaluated from another — the process aborts with `There is no Stream(gpu, N) in current thread`. So *every* MLX call (both `load()`s and all inference) goes through the single-thread `MlxWorker` in `app/core/mlx_worker.py` (`await worker.run(fn, ...)`). Never call `transcriber.transcribe` / `embedder.embed` / `load` directly from the event loop or `asyncio.to_thread`. ffmpeg is not MLX and may use `asyncio.to_thread`.

**Concurrency / busy handling:** `JobGate` in `app/core/gate.py` admits one GPU job at a time with a small bounded queue (`MAX_QUEUED_JOBS`, `QUEUE_WAIT_SECONDS`). `reserve()` runs at the start of a request and fails fast (503 `ServiceBusyError` + `Retry-After`) when the pipeline is full or the running job is estimated to outlast the wait budget; `run()` holds the GPU slot for the model call. The estimate uses the extracted WAV's size (`wav_duration_seconds`) and `TRANSCRIBE_SPEED_FACTOR`. `/health` exposes `busy`, `active_jobs`, `queued_jobs`, `estimated_wait_seconds`.

**Temp file naming:** Both endpoints use `{stem}_extracted.wav` for the ffmpeg output to avoid in-place overwrites when the input is already `.wav`.

## Key Conventions

- **Config:** All settings via Pydantic `BaseSettings` in `app/config.py`, driven by env vars / `.env` file. Use `_env_file=None` in tests to avoid loading `.env`.
- **Formatting:** Black (line-length 88, target py312). Ruff for linting (E, F, I rules). `known-first-party = ["app"]` for isort.
- **Testing:** pytest-asyncio with `asyncio_mode = "auto"` — async tests don't need `@pytest.mark.asyncio`. Mock mlx-whisper via `patch.dict(sys.modules, {"mlx_whisper": mock})` since it's imported inside methods. Integration tests patch `deps._transcriber`, `deps._media_service`, `deps._embedder` globals directly and set `deps._gate` to `deps.build_gate(settings)` so each test app gets a fresh gate (asyncio primitives must not leak across event loops). Busy-path tests block the mocked transcriber on a `threading.Event`.
- **File validation:** Upload endpoint: extension whitelist + magic bytes + size limit. URL endpoint: size limit only (ffmpeg handles format detection).
