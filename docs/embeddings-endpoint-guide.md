# Building an OpenAI-Compatible Embeddings Endpoint on Apple Silicon (MLX)

A porting guide for recreating the `POST /api/v1/embeddings` endpoint from
**whisperapy-mac** in another FastAPI project. It covers the design, every
file involved (with full code), configuration, error handling, the GPU
concurrency model, tests, and the pitfalls hit along the way.

- **Stack:** Python 3.12 · FastAPI · Pydantic v2 / pydantic-settings · `mlx-embeddings` · loguru
- **Model:** `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` (Qwen3-Embedding, 4-bit, 2560-dim output)
- **Hardware:** Apple Silicon only (Metal GPU). It cannot run in Docker, which has no Metal access.

---

## 1. What you are building

```
POST /api/v1/embeddings
Authorization: Bearer <API_KEY>        (optional; only if API_KEY is set)
Content-Type: application/json

{
  "input": "text" | ["text", ...],
  "model": "ignored-but-accepted",
  "encoding_format": "float" | "base64",
  "dimensions": 1024,                   // optional, Matryoshka truncation
  "prompt": "Instruct: ...\nQuery: "    // extension: prefix for query-side embeddings
}
```

Response (OpenAI shape, plus one extension field):

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.0123, ...]}],
  "model": "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
  "usage": {"prompt_tokens": 12, "total_tokens": 12},
  "truncated": 0
}
```

Because the endpoint matches OpenAI's, the official OpenAI SDK works without changes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/api/v1", api_key="...")
client.embeddings.create(model="qwen3", input=["hello"])
```

### Design goals

1. **OpenAI compatibility.** Accept the same request and return the same response, including the `base64` float32 encoding.
2. **Never crash the process with MLX thread errors.** All MLX work runs on a single dedicated thread.
3. **Never block the event loop.** Health checks and busy rejections must stay responsive while the GPU is working.
4. **Fail fast under load.** A bounded queue plus a `503` with `Retry-After` is better than an unbounded pile-up.
5. **Bounded GPU memory.** Large requests are split into fixed-size batches.
6. **Predictable limits.** Too many inputs or too many tokens return `422`, never `500`.

---

## 2. Architecture and request flow

```
Client
  │
  ▼
RequestContextMiddleware         (X-Request-ID, X-Processing-Time, loguru context)
  │
  ▼
require_api_key (router dep)     401 if API_KEY set and key wrong/missing
  │
  ▼
Pydantic EmbeddingRequest        422 on empty input / bad dims / bad format
  │
  ▼
create_embeddings()
  ├─ len(texts) > EMBEDDING_MAX_BATCH ─────────────► 422
  ├─ asyncio.to_thread(embedder.count_tokens)        CPU tokenizer, off the GPU thread
  ├─ gate.estimate_embed(sum(tokens))                seconds = tokens / EMBED_TOKENS_PER_SECOND
  ├─ async with gate.reserve("embed")  ────────────► 503 + Retry-After if pipeline full
  │    async with gate.run(job, estimate) ─────────► 503 if GPU wait > QUEUE_WAIT_SECONDS
  │       await worker.run(embedder.embed, ...)      ◄── runs on the single MLX thread
  │           ├─ prefix prompt
  │           ├─ count tokens again, enforce EMBEDDING_MAX_TOKENS ► 422 (or truncate)
  │           ├─ generate() in chunks of EMBEDDING_BATCH_SIZE
  │           └─ optional Matryoshka shrink + re-normalise
  └─ encode each vector (float list or base64 <f32) ─► 200 EmbeddingResponse
```

### File layout

```
app/
├── config.py                 # Settings (pydantic-settings)
├── dependencies.py           # singletons + Annotated DI aliases
├── main.py                   # lifespan: create services, load model on MLX thread
├── api/
│   ├── health.py             # /health reports embedder readiness
│   └── v1/
│       ├── router.py         # /api/v1 prefix + API-key dependency
│       ├── embeddings.py     # the endpoint
│       └── models.py         # GET /api/v1/models (for SDK probing)
├── core/
│   ├── exceptions.py         # domain errors carrying their own status code
│   ├── error_handler.py      # one handler -> {error, message, request_id}
│   ├── gate.py               # JobGate: GPU admission control
│   ├── mlx_worker.py         # single-thread executor for all MLX calls
│   └── security.py           # optional API key
├── schemas/embedding.py      # request/response models
└── services/embedder.py      # EmbedderService (load / count / embed)
```

---

## 3. Dependencies

`pyproject.toml`:

```toml
[project]
requires-python = ">=3.12"
dependencies = [
  "fastapi",
  "uvicorn[standard]",
  "pydantic>=2",
  "pydantic-settings",
  "loguru",
  "mlx-embeddings>=0.1.0",
]

[tool.pyright]
reportMissingImports = false   # mlx_embeddings is imported lazily inside methods
```

`mlx_embeddings` is imported **inside** methods, never at module top level. Tests can then replace it with `patch.dict(sys.modules, ...)`, and the test suite runs on machines without MLX, such as Linux CI.

---

## 4. The critical constraint: MLX thread affinity

> MLX keeps GPU streams **per thread**. A model loaded on thread A and evaluated
> on thread B aborts the whole process with
> `There is no Stream(gpu, N) in current thread`.

This rules out several approaches:
- ❌ Loading in the lifespan on the event loop, then running inference in `asyncio.to_thread` (a different pool thread each time)
- ❌ Running inference directly in an `async def` endpoint (blocks the event loop, and is still a different thread from the load)
- ✅ One `ThreadPoolExecutor(max_workers=1)` that runs **both** `load()` and every `embed()`

CPU-only work, such as the HF tokenizer's `encode`, is not MLX. It can and should go through `asyncio.to_thread` so it doesn't queue behind a long GPU job.

### `app/core/mlx_worker.py`

```python
"""A single dedicated thread for every MLX call."""

import asyncio
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


class MlxWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    async def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run ``fn`` on the MLX thread without blocking the event loop.

        The caller's contextvars (e.g. loguru's request_id) are copied onto the
        worker thread so log lines from inside the model call keep the id.
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._executor, functools.partial(ctx.run, fn, *args, **kwargs)
        )

    def run_sync(self, fn: Callable[..., T], *args, **kwargs) -> T:
        return self._executor.submit(functools.partial(fn, *args, **kwargs)).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
```

`loop.run_in_executor` does **not** copy contextvars on its own, unlike `asyncio.to_thread`. That is why the call is wrapped in `ctx.run`.

---

## 5. Configuration — `app/config.py`

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Security
    api_key: str | None = None

    # Embedding model
    embedding_model_repo: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
    # Inputs longer than this are rejected with 422 (OpenAI behaviour) unless
    # ``embedding_truncate`` is true, in which case they are cut and flagged.
    embedding_max_tokens: int = 8192
    embedding_truncate: bool = False
    # Max strings per request, and how many go to the GPU per forward pass.
    embedding_max_batch: int = 2048
    embedding_batch_size: int = 32

    # Concurrency (GPU runs one model call at a time)
    max_queued_jobs: int = 1
    queue_wait_seconds: float = 15.0
    # Rough embedding throughput; drives Retry-After while an embed job runs.
    embed_tokens_per_second: float = 2000.0
```

`.env.example`:

```dotenv
API_KEY=
EMBEDDING_MODEL_REPO=mlx-community/Qwen3-Embedding-4B-4bit-DWQ
EMBEDDING_MAX_TOKENS=8192   # longer inputs -> 422, or cut when EMBEDDING_TRUNCATE=true
EMBEDDING_TRUNCATE=false
EMBEDDING_MAX_BATCH=2048    # max strings per request
EMBEDDING_BATCH_SIZE=32     # strings per GPU forward pass
MAX_QUEUED_JOBS=1
QUEUE_WAIT_SECONDS=15
EMBED_TOKENS_PER_SECOND=2000
```

| Setting | Why it exists |
|---|---|
| `embedding_max_tokens` | Qwen3-Embedding supports long contexts, but memory and latency grow with length. 8192 is a practical cap. |
| `embedding_truncate` | OpenAI rejects over-long input, and silent truncation hides data loss. When enabled, truncation is reported in the `truncated` field. |
| `embedding_max_batch` | Rejects abusive requests before any work is done. It matches OpenAI's 2048-input limit. |
| `embedding_batch_size` | Caps the padded tensor per forward pass: `batch × max_len`. Lower it if memory is tight. |
| `embed_tokens_per_second` | Used only for estimates, which drive `Retry-After` and wait decisions. Measure it on your machine and tune it. |

`extra="ignore"` keeps boot from failing when `.env` contains keys that are no longer used.

---

## 6. Errors — `app/core/exceptions.py` and `app/core/error_handler.py`

Each domain error declares its own HTTP status, so the handler needs no lookup table.

```python
from typing import ClassVar


class AppError(Exception):
    status_code: ClassVar[int] = 500
    default_message: ClassVar[str] = "An error occurred"

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class AuthenticationError(AppError):
    status_code = 401
    default_message = "Missing or invalid API key"


class InvalidRequestError(AppError):
    """Request is well-formed but violates a configured limit."""
    status_code = 422
    default_message = "Invalid request"


class InputTooLongError(AppError):
    status_code = 422
    default_message = "Input exceeds the maximum token length"


class EmbeddingError(AppError):
    status_code = 500
    default_message = "Embedding failed"


class ModelNotReadyError(AppError):
    status_code = 503
    default_message = "Model is not ready"


class ServiceBusyError(AppError):
    status_code = 503
    default_message = "Server is busy processing another request. Please retry later."

    def __init__(self, message: str | None = None, retry_after: int = 30):
        self.retry_after = retry_after
        super().__init__(message)
```

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.exceptions import AppError, ServiceBusyError


def error_response(request: Request, exc: AppError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    content: dict = {"error": type(exc).__name__, "message": exc.message,
                     "request_id": request_id}
    headers: dict[str, str] = {}
    if isinstance(exc, ServiceBusyError):
        content["retry_after_seconds"] = exc.retry_after
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(status_code=exc.status_code, content=content, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(f"{type(exc).__name__}: {exc.message}")
        return error_response(request, exc)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(f"Unhandled error: {exc}")
        # Runs in Starlette's outermost ServerErrorMiddleware, outside our
        # request-context middleware, so set the header by hand.
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError",
                     "message": "An unexpected error occurred",
                     "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
```

Stack traces are never returned to the client.

---

## 7. Schemas — `app/schemas/embedding.py`

```python
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    input: str | list[str]
    model: str | None = None  # echoed/ignored; the server's model is authoritative
    encoding_format: Literal["float", "base64"] = "float"
    # Matryoshka truncation of the returned vector (Qwen3-Embedding supports it).
    dimensions: int | None = Field(default=None, ge=32, le=4096)
    # Extension: optional Qwen3 instruction prepended to every input.
    prompt: str | None = Field(default=None, max_length=2000)

    @field_validator("input")
    @classmethod
    def _reject_empty_input(cls, v: str | list[str]) -> str | list[str]:
        """Reject empty/blank input as a 422 rather than a 500 downstream."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("input must not be empty")
            return v
        if len(v) == 0:
            raise ValueError("input must not be an empty list")
        if any(not t.strip() for t in v):
            raise ValueError("input items must not be empty")
        return v

    @property
    def texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else self.input


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float] | str  # str when encoding_format == "base64"


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData] = Field(default_factory=list)
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    # Extension: inputs cut to EMBEDDING_MAX_TOKENS (0 unless truncation enabled).
    truncated: int = 0


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "my-service"
    task: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
```

OpenAI's API does **not** accept token-ID arrays here (`list[int]` / `list[list[int]]`). Add support only if a client needs it.

---

## 8. The model service — `app/services/embedder.py`

```python
import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from app.config import Settings
from app.core.exceptions import EmbeddingError, InputTooLongError, ModelNotReadyError


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int
    truncated: int  # number of inputs cut to embedding_max_tokens


class EmbedderService:
    """Singleton wrapper around mlx-embeddings (Qwen3-Embedding)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model_repo = settings.embedding_model_repo
        self._model: Any = None
        self._tokenizer: Any = None
        self._ready = False

    def load(self) -> None:
        """Load model + tokenizer. MUST be called on the MLX worker thread."""
        from mlx_embeddings import load

        logger.info(f"Loading embedding model: {self._model_repo}")
        start = time.perf_counter()
        self._model, self._tokenizer = load(self._model_repo)
        self._ready = True
        logger.info(f"Embedding model loaded in {time.perf_counter() - start:.1f}s")

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------ tokenizing

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Per-input token counts (CPU only, safe off the MLX thread)."""
        if self._tokenizer is None:
            return [0] * len(texts)
        try:
            return [len(self._tokenizer.encode(t)) for t in texts]
        except Exception:
            return [0] * len(texts)

    def check_lengths(self, counts: list[int]) -> int:
        """Enforce the token limit; return how many inputs will be truncated."""
        limit = self._settings.embedding_max_tokens
        over = [i for i, n in enumerate(counts) if n > limit]
        if over and not self._settings.embedding_truncate:
            raise InputTooLongError(
                f"Input {over[0]} is {counts[over[0]]} tokens; the limit is {limit}. "
                "Split the text or enable EMBEDDING_TRUNCATE."
            )
        return len(over)

    # -------------------------------------------------------------- embedding

    def embed(
        self,
        texts: list[str],
        prompt: str | None = None,
        dimensions: int | None = None,
    ) -> EmbedResult:
        """Embed a batch on the MLX thread, chunked by embedding_batch_size."""
        if not self._ready:
            raise ModelNotReadyError()

        from mlx_embeddings import generate

        inputs = [f"{prompt}{t}" for t in texts] if prompt else list(texts)
        counts = self.count_tokens(inputs)
        truncated = self.check_lengths(counts)
        max_len = self._settings.embedding_max_tokens
        batch = max(1, self._settings.embedding_batch_size)
        start = time.perf_counter()

        vectors: list[list[float]] = []
        try:
            for i in range(0, len(inputs), batch):
                output: Any = generate(
                    self._model,
                    self._tokenizer,
                    texts=inputs[i : i + batch],
                    max_length=max_len,
                    truncation=True,
                )
                vectors.extend(output.text_embeds.tolist())
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

        if dimensions is not None:
            vectors = [_shrink(v, dimensions) for v in vectors]

        # Usage reflects what was embedded, not what was sent.
        prompt_tokens = sum(min(n, max_len) for n in counts)
        logger.info(
            f"Embedded {len(inputs)} text(s) | tokens={prompt_tokens} | "
            f"truncated={truncated} | "
            f"processing={round(time.perf_counter() - start, 3)}s"
        )
        return EmbedResult(vectors=vectors, prompt_tokens=prompt_tokens,
                           truncated=truncated)


def _shrink(vector: list[float], dimensions: int) -> list[float]:
    """Matryoshka truncation: keep the leading dims and re-normalise."""
    head = vector[:dimensions]
    norm = sum(x * x for x in head) ** 0.5
    return [x / norm for x in head] if norm else head
```

### Key points

- **`text_embeds`** is the pooled, L2-normalised sentence embedding that `mlx-embeddings` returns. For Qwen3 it uses last-token pooling. Because the vectors are unit length, cosine similarity equals the dot product.
- **Chunking** by `embedding_batch_size` prevents one request of 2048 strings from padding into a single huge tensor.
- **`truncation=True` is always passed** to `generate`, as a safety net. The *policy* (reject or truncate) is enforced earlier by `check_lengths`.
- **Matryoshka (`dimensions`)**: Qwen3-Embedding is trained so that its leading dimensions form a usable lower-dimensional embedding. After slicing, the vector must be **re-normalised**, otherwise similarity scores shift.
- **Qwen3 instruction prompts**: queries should carry an instruction and documents should not:
  - Query: `prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`
  - Document: no `prompt`

  This is why `prompt` is a per-request field and not a global setting.

---

## 9. GPU admission control — `app/core/gate.py`

The GPU runs one model call at a time. Without a gate, requests would queue without limit in the executor, clients would time out, and health checks would look fine while the backlog grew. `JobGate` provides:

- **`reserve(kind)`**: claims a slot in the pipeline, which is *running plus queued*. It fails **immediately** with 503 when the pipeline is full, or when the running job is estimated to outlast the wait budget.
- **`run(job, estimated_seconds)`**: waits up to `queue_wait_seconds` for the GPU semaphore and records the job's estimate, so *other* requests can compute `Retry-After`.

```python
import asyncio
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from loguru import logger

from app.core.exceptions import ServiceBusyError

_DEFAULT_RETRY_AFTER = 30     # when nothing is known about the running job
_MIN_RETRY_AFTER = 5          # never advertise less, so clients don't hammer
_JOB_OVERHEAD_SECONDS = 2.0   # fixed per-job overhead added to estimates


@dataclass
class Job:
    kind: str
    reserved_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    estimated_seconds: float | None = None

    @property
    def running(self) -> bool:
        return self.started_at is not None


class JobGate:
    def __init__(self, max_concurrent: int = 1, max_queued: int = 1,
                 queue_wait_seconds: float = 15.0,
                 embed_tokens_per_second: float = 2000.0):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.queue_wait_seconds = max(0.0, queue_wait_seconds)
        self.embed_tokens_per_second = max(1.0, embed_tokens_per_second)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: list[Job] = []

    def estimate_embed(self, tokens: int | None) -> float | None:
        return None if tokens is None else tokens / self.embed_tokens_per_second

    @property
    def active_jobs(self) -> int:
        return sum(1 for j in self._jobs if j.running)

    @property
    def queued_jobs(self) -> int:
        return sum(1 for j in self._jobs if not j.running)

    @property
    def busy(self) -> bool:
        return self.active_jobs >= self.max_concurrent

    def estimated_wait_seconds(self) -> float | None:
        if not self.busy:
            return None
        now = time.monotonic()
        remaining = [
            max(0.0, j.estimated_seconds + _JOB_OVERHEAD_SECONDS
                - (now - (j.started_at or now)))
            for j in self._jobs
            if j.running and j.estimated_seconds is not None
        ]
        return min(remaining) if remaining else None

    def retry_after(self) -> int:
        if not self.busy:
            return _MIN_RETRY_AFTER
        est = self.estimated_wait_seconds()
        if est is None:
            return _DEFAULT_RETRY_AFTER
        return max(_MIN_RETRY_AFTER, math.ceil(est))

    def snapshot(self) -> dict:
        est = self.estimated_wait_seconds()
        return {
            "busy": self.busy,
            "active_jobs": self.active_jobs,
            "queued_jobs": self.queued_jobs,
            "max_concurrent_jobs": self.max_concurrent,
            "max_queued_jobs": self.max_queued,
            "estimated_wait_seconds": None if est is None else round(est),
        }

    def _reject(self, reason: str, kind: str) -> ServiceBusyError:
        retry_after = self.retry_after()
        logger.warning(f"Rejecting {kind} job: {reason} | retry_after={retry_after}s")
        return ServiceBusyError(retry_after=retry_after)

    def _wait_is_hopeless(self) -> bool:
        est = self.estimated_wait_seconds()
        return est is not None and est > self.queue_wait_seconds

    @asynccontextmanager
    async def reserve(self, kind: str) -> AsyncIterator[Job]:
        if len(self._jobs) >= self.max_concurrent + self.max_queued:
            raise self._reject("pipeline full", kind)
        if self.busy and self._wait_is_hopeless():
            raise self._reject("running job exceeds wait budget", kind)
        job = Job(kind=kind)
        self._jobs.append(job)
        try:
            yield job
        finally:
            self._jobs.remove(job)

    @asynccontextmanager
    async def run(self, job: Job,
                  estimated_seconds: float | None = None) -> AsyncIterator[None]:
        if self.busy and self._wait_is_hopeless():
            raise self._reject("running job exceeds wait budget", job.kind)
        try:
            await asyncio.wait_for(self._semaphore.acquire(),
                                   timeout=self.queue_wait_seconds)
        except TimeoutError:
            raise self._reject("queue wait timed out", job.kind) from None
        job.started_at = time.monotonic()
        job.estimated_seconds = estimated_seconds
        try:
            yield
        finally:
            job.started_at = None
            self._semaphore.release()
```

`max_concurrent` is always 1, because there is a single MLX thread. If the other project runs several models, such as whisper plus embeddings, **share one gate** between them, because they share one GPU and one thread.

---

## 10. Dependency injection — `app/dependencies.py`

```python
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings
from app.core.gate import JobGate
from app.core.mlx_worker import MlxWorker
from app.services.embedder import EmbedderService

_embedder: EmbedderService | None = None
_gate: JobGate | None = None
_mlx_worker: MlxWorker | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_embedder() -> EmbedderService:
    if _embedder is None:
        raise RuntimeError("EmbedderService not initialized")
    return _embedder


def get_gate() -> JobGate:
    global _gate
    if _gate is None:          # lazy so tests work without the lifespan
        _gate = build_gate(get_settings())
    return _gate


def get_mlx_worker() -> MlxWorker:
    global _mlx_worker
    if _mlx_worker is None:
        _mlx_worker = MlxWorker()
    return _mlx_worker


def build_gate(settings: Settings) -> JobGate:
    return JobGate(
        max_concurrent=1,
        max_queued=settings.max_queued_jobs,
        queue_wait_seconds=settings.queue_wait_seconds,
        embed_tokens_per_second=settings.embed_tokens_per_second,
    )


def init_services(settings: Settings) -> None:
    global _embedder, _gate, _mlx_worker
    _embedder = EmbedderService(settings)
    _gate = build_gate(settings)
    _mlx_worker = MlxWorker()


# Annotated aliases: no ``= Depends(...)`` defaults (ruff B008).
SettingsDep = Annotated[Settings, Depends(get_settings)]
EmbedderDep = Annotated[EmbedderService, Depends(get_embedder)]
GateDep = Annotated[JobGate, Depends(get_gate)]
WorkerDep = Annotated[MlxWorker, Depends(get_mlx_worker)]
```

---

## 11. The endpoint — `app/api/v1/embeddings.py`

```python
import asyncio
import base64
import struct

from fastapi import APIRouter

from app.core.exceptions import InvalidRequestError
from app.dependencies import EmbedderDep, GateDep, SettingsDep, WorkerDep
from app.schemas.embedding import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)

router = APIRouter(prefix="/embeddings", tags=["embeddings", "openai-compatible"])


def _encode(vector: list[float], fmt: str) -> list[float] | str:
    if fmt == "base64":
        # OpenAI encodes float32 little-endian.
        return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode()
    return vector


@router.post("", response_model=EmbeddingResponse)
async def create_embeddings(
    body: EmbeddingRequest,
    settings: SettingsDep,
    embedder: EmbedderDep,
    gate: GateDep,
    worker: WorkerDep,
) -> EmbeddingResponse:
    """OpenAI-compatible text embeddings via Qwen3-Embedding."""
    texts = body.texts
    if len(texts) > settings.embedding_max_batch:
        raise InvalidRequestError(
            f"input has {len(texts)} items; the limit is {settings.embedding_max_batch}"
        )

    # Token counting is CPU-only (HF tokenizer, not MLX), so it runs in an
    # ordinary thread rather than queueing behind whatever holds the GPU. It
    # happens before the slot is taken so the gate can estimate this job.
    counts = await asyncio.to_thread(embedder.count_tokens, texts)
    estimate = gate.estimate_embed(sum(counts))

    async with gate.reserve("embed") as job, gate.run(job, estimated_seconds=estimate):
        result = await worker.run(
            embedder.embed, texts, prompt=body.prompt, dimensions=body.dimensions
        )

    data = [
        EmbeddingData(index=i, embedding=_encode(vec, body.encoding_format))
        for i, vec in enumerate(result.vectors)
    ]
    return EmbeddingResponse(
        data=data,
        model=settings.embedding_model_repo,
        usage=Usage(prompt_tokens=result.prompt_tokens,
                    total_tokens=result.prompt_tokens),
        truncated=result.truncated,
    )
```

Base64 details: the OpenAI Python SDK sends `encoding_format="base64"` **by default** and decodes the result with `np.frombuffer(base64.b64decode(s), dtype="float32")`. Any other byte layout, such as float64 or big-endian, returns incorrect vectors with no error. Keep `"<f"`.

---

## 12. Router, API key, models listing

`app/api/v1/router.py`:

```python
from fastapi import APIRouter, Depends

from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router
from app.core.security import require_api_key

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])
router.include_router(embeddings_router)
router.include_router(models_router)
```

`app/core/security.py`: accepts `Bearer` or `X-API-Key` and uses a constant-time comparison.

```python
import hmac

from fastapi import Request

from app.core.exceptions import AuthenticationError
from app.dependencies import SettingsDep


def _presented_key(request: Request) -> str | None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-api-key")


async def require_api_key(request: Request, settings: SettingsDep) -> None:
    if settings.api_key is None:
        return
    presented = _presented_key(request)
    if presented is None or not hmac.compare_digest(presented, settings.api_key):
        raise AuthenticationError()
```

`app/api/v1/models.py`: some SDKs and tools probe `/models` on startup.

```python
from fastapi import APIRouter

from app.dependencies import SettingsDep
from app.schemas.embedding import ModelInfo, ModelList

router = APIRouter(prefix="/models", tags=["openai-compatible"])


@router.get("", response_model=ModelList)
async def list_models(settings: SettingsDep) -> ModelList:
    return ModelList(data=[ModelInfo(id=settings.embedding_model_repo, task="embedding")])
```

---

## 13. Startup and health — `app/main.py`, `app/api/health.py`

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health
from app.api.v1.router import router as v1_router
from app.core.error_handler import register_error_handlers
from app.dependencies import get_embedder, get_mlx_worker, get_settings, init_services


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    init_services(settings)
    worker = get_mlx_worker()
    # Load on the MLX thread — the SAME thread that will run inference.
    # A load failure propagates and aborts startup on purpose.
    await worker.run(get_embedder().load)
    yield
    worker.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(health.router)     # outside /api/v1 -> never behind the key
    app.include_router(v1_router)
    return app


app = create_app()
```

Health endpoint, which returns 503 until the model is loaded so load balancers and launchd scripts can wait for it:

```python
from fastapi import APIRouter, Response

from app import dependencies as deps
from app.dependencies import SettingsDep

router = APIRouter()


@router.get("/health")
async def health(settings: SettingsDep, response: Response) -> dict:
    ready = deps._embedder is not None and deps._embedder.is_ready()
    if not ready:
        response.status_code = 503
    return {
        "status": "ok" if ready else "starting",
        "embedding_model": settings.embedding_model_repo.split("/")[-1],
        "embedding_model_loaded": ready,
        **deps.get_gate().snapshot(),    # busy / active_jobs / queued_jobs / wait
    }
```

Optional: warm up once after `load()` by running `generate(model, tok, texts=["warmup"])` on the MLX thread. This compiles the Metal kernels, so the first real request isn't slow. The whisper side of this project does that; the embedder currently does not.

---

## 14. Error reference

All errors use the body shape `{"error": "<ClassName>", "message": "...", "request_id": "..."}`.

| Status | Error | Trigger |
|---|---|---|
| 401 | `AuthenticationError` | `API_KEY` set, and key missing or wrong |
| 422 | (FastAPI validation) | empty input, blank item, `dimensions` ∉ [32, 4096], bad `encoding_format`, `prompt` > 2000 chars |
| 422 | `InvalidRequestError` | more than `EMBEDDING_MAX_BATCH` inputs |
| 422 | `InputTooLongError` | an input exceeds `EMBEDDING_MAX_TOKENS` and truncation is off |
| 500 | `EmbeddingError` | `generate()` raised an exception |
| 503 | `ModelNotReadyError` | model not loaded yet |
| 503 | `ServiceBusyError` | pipeline full or GPU wait too long; adds a `Retry-After` header and `retry_after_seconds` |

---

## 15. Tests

Use pytest with `pytest-asyncio` (`asyncio_mode = "auto"`) and `httpx.AsyncClient` with `ASGITransport`.

### Unit: service with a fake `mlx_embeddings`

```python
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.core.exceptions import InputTooLongError
from app.services.embedder import EmbedderService


@pytest.fixture
def mock_mlx_embeddings():
    mock = MagicMock()
    with patch.dict(sys.modules, {"mlx_embeddings": mock}):
        yield mock


def make_service(**overrides) -> EmbedderService:
    svc = EmbedderService(Settings(_env_file=None, **overrides))
    svc._ready = True
    svc._tokenizer = MagicMock()
    # one token per word keeps the arithmetic obvious
    svc._tokenizer.encode.side_effect = lambda t: list(range(len(t.split())))
    return svc


def set_output(mock, dims=2):
    def gen(model, tokenizer, texts, **kwargs):
        out = MagicMock()
        out.text_embeds.tolist.return_value = [[0.1] * dims for _ in texts]
        return out
    mock.generate.side_effect = gen


def test_rejects_over_length_input(mock_mlx_embeddings):
    svc = make_service(embedding_max_tokens=2)
    set_output(mock_mlx_embeddings)
    with pytest.raises(InputTooLongError, match="Input 1 is 3 tokens"):
        svc.embed(["ok ok", "one two three"])
    mock_mlx_embeddings.generate.assert_not_called()
```

Service tests worth having:
- not ready → `ModelNotReadyError`
- `load()` sets ready and calls `load(repo)`
- usage counts tokens actually embedded
- `max_length` and `truncation=True` are forwarded to `generate`
- over-length input is rejected by default, and truncated and counted when `embedding_truncate=True`
- chunking: 70 inputs with `batch_size=32` → 3 `generate` calls
- the `prompt` is prefixed to each input
- `_shrink` returns a unit vector of the requested length
- a `generate` exception is wrapped as `EmbeddingError`

### Integration: endpoint with a mocked service

Patch the module-level singletons and override `get_settings`:

```python
embedder = MagicMock()
embedder.count_tokens.side_effect = lambda texts: [5 for _ in texts]
embedder.embed.side_effect = lambda texts, prompt=None, dimensions=None: EmbedResult(
    vectors=[[0.1, 0.2, 0.3] for _ in texts], prompt_tokens=5 * len(texts), truncated=0
)
with patch.object(deps, "_embedder", embedder), \
     patch.object(deps, "_gate", deps.build_gate(settings)):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ...
```

Integration tests worth having:
- list input and string input → correct `data`, `index`, `usage`
- `base64` decodes with `struct.unpack("<3f", ...)`
- unknown `encoding_format`, empty input, or `dimensions` of 1 or 10000 → 422
- `prompt` and `dimensions` are forwarded to `embed`
- more than `embedding_max_batch` inputs → 422
- API key required, and accepted through both header styles
- **busy path:** block the mocked `embed` on a `threading.Event`, send a second request, and assert `503` plus a `Retry-After` header

---

## 16. Pitfalls and lessons learned

1. **MLX thread affinity.** Load and infer on the same single thread (see §4). This is the most important rule, and breaking it aborts the process rather than raising an exception.
2. **Don't tokenize on the MLX thread before admission.** Counting is CPU work. If it ran on the MLX thread, a simple token count would wait behind a 10-minute transcription.
3. **Contextvars across `run_in_executor`.** Without `copy_context()`, request IDs vanish from model-side logs.
4. **The base64 format must be float32 little-endian.** The SDK's default path depends on it.
5. **Re-normalise after Matryoshka truncation.** Otherwise similarity scores are silently wrong.
6. **Report real usage.** Count the tokens actually embedded (capped at the limit), including the prompt prefix.
7. **Lazy imports of `mlx_embeddings`.** They keep tests and type checking independent of the hardware.
8. **Health returns 503 until loaded, and model-load failures crash startup.** A server that cannot embed should never report healthy.

### Improvements to make when porting

These are known gaps in the current implementation that are cheap to fix in a new project:

- **Check lengths before taking the GPU slot.** The endpoint already has `counts`, so call `embedder.check_lengths(counts)` right after `count_tokens` and the 422 is returned without queueing. Include the prompt in that count by counting `f"{prompt}{t}"`, which also makes the gate estimate exact:

  ```python
  prefixed = [f"{body.prompt}{t}" for t in texts] if body.prompt else texts
  counts = await asyncio.to_thread(embedder.count_tokens, prefixed)
  embedder.check_lengths(counts)                 # 422 before reserve()
  estimate = gate.estimate_embed(sum(counts))
  ```

- **Reject a `dimensions` value above the model's width.** Qwen3-Embedding-4B outputs 2560 dimensions, but the schema allows up to 4096, and anything above 2560 silently returns 2560. Read the width after `load()` (for example by embedding `"x"` once during warm-up) and raise `InvalidRequestError` when `dimensions > width`.
- **Don't treat a tokenizer failure as zero tokens.** When `count_tokens` fails it returns 0, which skips the limit check and reports 0 usage. It is better to raise, or to fall back to an estimate such as `len(text) // 3`.
- **Add a warm-up embed after load** so Metal kernel compilation doesn't slow the first real request.

---

## 17. Smoke test

```bash
uv run uvicorn app.main:app --port 8000
curl -s localhost:8000/health | jq

curl -s localhost:8000/api/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"input": ["hello world", "second doc"], "dimensions": 1024}' \
  | jq '.data[0].embedding | length, .usage'

# Query-side embedding with the Qwen3 instruction prefix
curl -s localhost:8000/api/v1/embeddings -H "Content-Type: application/json" -d '{
  "input": "how do I reset my password?",
  "prompt": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
}' | jq '.usage'
```
