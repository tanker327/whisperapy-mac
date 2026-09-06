import threading

from app.core.mlx_worker import MlxWorker


async def test_all_calls_share_one_thread_off_the_event_loop():
    worker = MlxWorker()
    try:
        names = {await worker.run(lambda: threading.current_thread().name)}
        names.add(await worker.run(lambda: threading.current_thread().name))
        names.add(worker.run_sync(lambda: threading.current_thread().name))
        assert len(names) == 1
        assert names.pop().startswith("mlx")
        assert names != {threading.current_thread().name}
    finally:
        worker.shutdown()


async def test_run_passes_args_and_kwargs():
    worker = MlxWorker()
    try:
        assert await worker.run(lambda a, b=0: a + b, 2, b=3) == 5
    finally:
        worker.shutdown()
