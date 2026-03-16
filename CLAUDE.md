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

**API:** `POST /api/v1/transcribe` (sync, multipart upload), `POST /api/v1/transcribe/url` (sync, JSON body with URL), `GET /health`, `GET /health/model`. Default model: `mlx-community/whisper-large-v3-turbo`.

**Request flow (upload):** Client → Middleware (request ID, timing) → Endpoint → `file_handler.validate_upload` → `MediaService.extract_audio` (ffmpeg → 16kHz WAV) → `TranscriberService.transcribe` (mlx-whisper) → `cleanup_temp` → Response

**Request flow (URL):** Client → Middleware → Endpoint → `file_handler.download_file_from_url` (httpx streaming + size limit) → `MediaService.extract_audio` → `TranscriberService.transcribe` → `cleanup_temp` → Response. No format/extension validation — ffmpeg handles whatever it gets.

**Dependency injection:** Services are module-level singletons in `app/dependencies.py`. `init_services()` is called once during FastAPI lifespan startup. Endpoints inject via `Depends(get_transcriber)`, `Depends(get_media_service)`, `Depends(get_settings)`. Settings use `@lru_cache`.

**Error handling:** All domain errors extend `WhisperapyError` (in `app/core/exceptions.py`). Global handlers in `app/core/error_handler.py` map each subclass to an HTTP status code and return consistent `{error, message, request_id}` JSON. Stack traces are never exposed.

**Model lifecycle:** mlx-whisper model loads once at startup via lifespan context manager in `app/main.py`. All requests share the singleton — no per-request load cost. `mlx_whisper` is imported inside methods (not at top-level) to avoid import errors on non-Apple-Silicon machines and in tests.

## Key Conventions

- **Config:** All settings via Pydantic `BaseSettings` in `app/config.py`, driven by env vars / `.env` file. Use `_env_file=None` in tests to avoid loading `.env`.
- **Formatting:** Black (line-length 88, target py312). Ruff for linting (E, F, I rules). `known-first-party = ["app"]` for isort.
- **Testing:** pytest-asyncio with `asyncio_mode = "auto"` — async tests don't need `@pytest.mark.asyncio`. Mock mlx-whisper via `patch.dict(sys.modules, {"mlx_whisper": mock})` since it's imported inside methods. Integration tests patch `deps._transcriber` and `deps._media_service` globals directly.
- **File validation:** Extension whitelist + magic bytes verification + size limit. See `MAGIC_BYTES` dict in `app/utils/file_handler.py`.
- **Async jobs:** Not yet implemented (v2). Only the sync `POST /api/v1/transcribe` endpoint is active.
