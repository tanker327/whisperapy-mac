"""A single dedicated thread for every MLX call.

MLX (0.32+) keeps GPU streams per thread. A model loaded or warmed up on one
thread cannot be safely evaluated from another — doing so aborts the whole
process with ``There is no Stream(gpu, N) in current thread``. So model
loading *and* inference for both mlx-whisper and mlx-embeddings must happen on
one thread, and that thread must not be the event loop.

``MlxWorker`` owns that thread. Endpoints ``await worker.run(fn, ...)`` so the
event loop stays free to serve health checks and reject busy requests.
"""

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")


class MlxWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

    async def run(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run ``fn`` on the MLX thread without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, functools.partial(fn, *args, **kwargs)
        )

    def run_sync(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Run ``fn`` on the MLX thread and block until it returns."""
        return self._executor.submit(functools.partial(fn, *args, **kwargs)).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
