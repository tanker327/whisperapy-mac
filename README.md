# whisperapy-mac

Local REST transcription **and embedding** service powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) on Apple Silicon. No cloud, no usage fees, full privacy.

- **Mac native** — direct Metal GPU access via MLX (no Docker)
- **Fast** — `whisper-large-v3-turbo` runs ~8x faster than base, using only ~1.5 GB of unified memory
- **Any format** — accepts video and audio files (mp4, mov, mkv, avi, webm, mp3, wav, m4a, ogg, flac, aac)
- **Embeddings** — OpenAI-compatible `/api/v1/embeddings` endpoint backed by `Qwen3-Embedding-4B-4bit-DWQ` (drop-in for OpenAI SDK / LangChain clients)
- **Production-grade** — structured logging, request tracing, global error handling

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (`brew install ffmpeg`)

## Quick Start

```bash
# Install dependencies
make install-dev

# Copy and configure environment
cp .env.example .env

# Start the server (downloads models on first run: whisper ~1.5 GB + Qwen3-Embedding-4B-4bit-DWQ ~2 GB)
make dev
```

The server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## API

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "whisper-large-v3-turbo",
  "model_loaded": true,
  "embedding_model": "Qwen3-Embedding-4B-4bit-DWQ",
  "embedding_model_loaded": true,
  "uptime_seconds": 3600
}
```

### Transcribe

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@recording.mp4" \
  -F "language=auto" \
  -F "word_timestamps=false"
```

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

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | file | required | Video or audio file |
| `language` | string | `auto` | Language code (e.g. `en`, `zh`) or `auto` to detect |
| `word_timestamps` | bool | `false` | Include word-level timestamps |
| `output_format` | string | `json` | `json`, `text`, `srt`, or `vtt` |

### Embeddings

OpenAI-compatible. Accepts a single string or a list of strings as `input`.

```bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "goodbye"]}'
```

```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.01, -0.02, "..."] },
    { "object": "embedding", "index": 1, "embedding": [0.03, -0.04, "..."] }
  ],
  "model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
  "usage": { "prompt_tokens": 6, "total_tokens": 6 }
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string \| string[] | required | Text (or list of texts) to embed |
| `model` | string | `null` | Echoed back; the server-configured model is authoritative |
| `encoding_format` | string | `float` | Encoding of the returned vectors |
| `prompt` | string | `null` | Optional instruction prepended to each input (e.g. for Qwen3 query-side embeddings) |

## Development

```bash
make test       # Run tests
make lint       # Lint with ruff
make format     # Format with black
make check      # All three in sequence
make clean      # Remove tmp/ and __pycache__
```

## Configuration

All settings are configured via environment variables or `.env` file. See [`.env.example`](.env.example) for available options.

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug logging |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `MODEL_REPO` | `mlx-community/whisper-large-v3-turbo` | HuggingFace transcription model repo |
| `DEFAULT_LANGUAGE` | `auto` | Default language for transcription |
| `EMBEDDING_MODEL_REPO` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` | HuggingFace embedding model repo |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `TEMP_DIR` | `./tmp` | Directory for temporary files |
| `MAX_QUEUED_JOBS` | `1` | Requests allowed to wait for the GPU; beyond this, 503 immediately |
| `QUEUE_WAIT_SECONDS` | `15` | Max time a queued request waits before returning 503 |
| `TRANSCRIBE_SPEED_FACTOR` | `8` | Assumed transcription speed vs. real time, used to estimate `Retry-After` |

### Concurrency and busy responses

The Metal GPU runs one model call at a time, so transcription and embedding
requests share a single job slot. Both models are loaded on, and every model
call runs on, one dedicated MLX thread (MLX keeps GPU streams per thread, so
loading and inference must share a thread). ffmpeg runs in a regular worker
thread. The event loop is never blocked, so `/health` and busy rejections
respond instantly during a long transcription.

When a request arrives while the GPU is busy:

- If the running job is expected to finish within `QUEUE_WAIT_SECONDS`, the
  request waits for the slot (at most `MAX_QUEUED_JOBS` may wait).
- Otherwise it fails immediately with **HTTP 503**, a `Retry-After` header,
  and a JSON body including `retry_after_seconds`. For URL requests this check
  runs before the download starts.

`GET /health` reports `busy`, `active_jobs`, `queued_jobs`, and
`estimated_wait_seconds` so clients can decide when to retry.

```json
{
  "error": "ServiceBusyError",
  "message": "Server is busy processing another request. Please retry later.",
  "request_id": "…",
  "retry_after_seconds": 42
}
```
