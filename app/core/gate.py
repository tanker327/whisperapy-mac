"""Admission control for GPU-bound work.

mlx-whisper and mlx-embeddings both run on the single Metal GPU and are not
safe (or fast) to run concurrently, so every model call goes through one
``JobGate``. The gate has two layers:

1. ``reserve()`` — called at the very start of a request, before any upload
   or download work. It fails immediately with ``ServiceBusyError`` when the
   pipeline is already full, or when the job currently on the GPU is
   estimated to run longer than the caller is willing to wait. This avoids
   downloading a large file only to discover the GPU is tied up.
2. ``run()`` — called right before the model call. It waits for a GPU slot
   for at most ``queue_wait_seconds`` and raises ``ServiceBusyError`` on
   timeout.

Every ``ServiceBusyError`` carries a ``retry_after`` estimate derived from the
expected length of the active job: audio seconds over a real-time speed factor
for transcription, token count over a throughput figure for embeddings.
"""

import asyncio
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from loguru import logger

from app.core.exceptions import ServiceBusyError

# Fallback estimate when nothing is known about the running job.
_DEFAULT_RETRY_AFTER = 30
# Minimum Retry-After we ever advertise, so clients don't hammer the server.
_MIN_RETRY_AFTER = 5
# Fixed per-job overhead (model warm-up, IO) added to every estimate.
_JOB_OVERHEAD_SECONDS = 2.0


@dataclass
class Job:
    kind: str
    reserved_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    # Expected wall-clock seconds of GPU work, or None when unknown.
    estimated_seconds: float | None = None

    @property
    def running(self) -> bool:
        return self.started_at is not None


class JobGate:
    def __init__(
        self,
        max_concurrent: int = 1,
        max_queued: int = 1,
        queue_wait_seconds: float = 15.0,
        speed_factor: float = 8.0,
        embed_tokens_per_second: float = 2000.0,
    ):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")
        self.max_concurrent = max_concurrent
        self.max_queued = max_queued
        self.queue_wait_seconds = max(0.0, queue_wait_seconds)
        self.speed_factor = max(0.1, speed_factor)
        self.embed_tokens_per_second = max(1.0, embed_tokens_per_second)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._jobs: list[Job] = []

    # -------------------------------------------------------------- estimates

    def estimate_transcribe(self, audio_seconds: float | None) -> float | None:
        if audio_seconds is None:
            return None
        return audio_seconds / self.speed_factor

    def estimate_embed(self, tokens: int | None) -> float | None:
        if tokens is None:
            return None
        return tokens / self.embed_tokens_per_second

    # ------------------------------------------------------------------ state

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
        """Best-effort seconds until a GPU slot frees up.

        Returns ``None`` when no slot is held or when nothing is known about
        the running jobs' length. Otherwise the smallest remaining time across
        running jobs with a known estimate.
        """
        if not self.busy:
            return None
        now = time.monotonic()
        remaining: list[float] = []
        for job in self._jobs:
            if not job.running or job.estimated_seconds is None:
                continue
            expected = job.estimated_seconds + _JOB_OVERHEAD_SECONDS
            elapsed = now - (job.started_at or now)
            remaining.append(max(0.0, expected - elapsed))
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

    # -------------------------------------------------------------- admission

    def _reject(self, reason: str, kind: str) -> ServiceBusyError:
        retry_after = self.retry_after()
        logger.warning(
            f"Rejecting {kind} job: {reason} | "
            f"active={self.active_jobs} queued={self.queued_jobs} "
            f"retry_after={retry_after}s"
        )
        return ServiceBusyError(retry_after=retry_after)

    def _wait_is_hopeless(self) -> bool:
        """True when the running job will clearly outlast our wait budget.

        Jobs of unknown length never count as hopeless; callers simply wait
        up to ``queue_wait_seconds`` for them.
        """
        est = self.estimated_wait_seconds()
        return est is not None and est > self.queue_wait_seconds

    @asynccontextmanager
    async def reserve(self, kind: str) -> AsyncIterator[Job]:
        """Claim a pipeline slot for the whole request. Fails fast when full."""
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
    async def run(
        self, job: Job, estimated_seconds: float | None = None
    ) -> AsyncIterator[None]:
        """Hold a GPU slot for the duration of the model call."""
        if self.busy and self._wait_is_hopeless():
            raise self._reject("running job exceeds wait budget", job.kind)

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.queue_wait_seconds
            )
        except TimeoutError:
            raise self._reject("queue wait timed out", job.kind) from None

        job.started_at = time.monotonic()
        job.estimated_seconds = estimated_seconds
        try:
            yield
        finally:
            job.started_at = None
            self._semaphore.release()
