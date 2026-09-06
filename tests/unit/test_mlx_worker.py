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


async def test_request_context_reaches_the_worker_thread(capsys):
    """Log lines from inside a model call must carry the request id."""
    from loguru import logger

    from app.config import Settings
    from app.core.logging import setup_logging

    setup_logging(Settings(debug=True, _env_file=None))
    worker = MlxWorker()
    try:
        with logger.contextualize(request_id="req-on-worker"):
            await worker.run(lambda: logger.info("from the mlx thread"))
    finally:
        worker.shutdown()
    line = next(
        line for line in capsys.readouterr().out.splitlines() if "mlx thread" in line
    )
    assert "req-on-worker" in line
    assert "no-request" not in line
