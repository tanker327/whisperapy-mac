# whisperapy-mac

Local REST transcription service powered by [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon. No cloud, no usage fees, full privacy.

- **Mac native** — direct Metal GPU access via MLX (no Docker)
- **Fast** — `whisper-large-v3-turbo` runs ~8x faster than base, using only ~1.5 GB of unified memory
- **Any format** — accepts video and audio files (mp4, mov, mkv, avi, webm, mp3, wav, m4a, ogg, flac, aac)
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

# Start the server (downloads model on first run, ~1.5 GB)
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
| `MODEL_REPO` | `mlx-community/whisper-large-v3-turbo` | HuggingFace model repo |
| `DEFAULT_LANGUAGE` | `auto` | Default language for transcription |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `TEMP_DIR` | `./tmp` | Directory for temporary files |
