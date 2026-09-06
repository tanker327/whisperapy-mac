import asyncio

import pytest

from app.core.exceptions import ServiceBusyError
from app.core.gate import JobGate


async def test_single_job_runs():
    gate = JobGate(max_concurrent=1, max_queued=1, queue_wait_seconds=1)
    async with gate.reserve("transcribe") as job:
        async with gate.run(job, audio_seconds=10):
            assert gate.busy
            assert gate.active_jobs == 1
    assert not gate.busy
    assert gate.active_jobs == 0
    assert gate.queued_jobs == 0


async def test_second_job_waits_then_runs():
    gate = JobGate(max_concurrent=1, max_queued=1, queue_wait_seconds=5)
    release = asyncio.Event()
    order: list[str] = []

    async def first():
        async with gate.reserve("a") as job, gate.run(job, audio_seconds=1):
            order.append("a-start")
            await release.wait()
            order.append("a-end")

    async def second():
        await asyncio.sleep(0.01)
        async with gate.reserve("b") as job:
            assert gate.queued_jobs == 1
            async with gate.run(job):
                order.append("b-start")

    t1 = asyncio.create_task(first())
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(t1, t2)
    assert order == ["a-start", "a-end", "b-start"]


async def test_pipeline_full_rejects_immediately():
    gate = JobGate(max_concurrent=1, max_queued=1, queue_wait_seconds=5)
    release = asyncio.Event()

    async def hold():
        async with gate.reserve("a") as job, gate.run(job):
            await release.wait()

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    async with gate.reserve("b"):  # fills the single queue slot
        with pytest.raises(ServiceBusyError) as exc:
            async with gate.reserve("c"):
                pass
        # Running job has unknown length => default estimate.
        assert exc.value.retry_after == 30
    release.set()
    await t


async def test_unknown_length_job_never_fast_rejects():
    """A job of unknown length must let callers wait, not fail instantly."""
    gate = JobGate(max_concurrent=1, max_queued=1, queue_wait_seconds=0.05)
    release = asyncio.Event()

    async def hold():
        async with gate.reserve("a") as job, gate.run(job):
            await release.wait()

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    async with gate.reserve("b"):
        assert gate.queued_jobs == 1
        assert gate.estimated_wait_seconds() is None
    release.set()
    await t


async def test_queue_wait_timeout_rejects():
    gate = JobGate(max_concurrent=1, max_queued=1, queue_wait_seconds=0.05)
    release = asyncio.Event()

    async def hold():
        async with gate.reserve("a") as job, gate.run(job):
            await release.wait()

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    with pytest.raises(ServiceBusyError):
        async with gate.reserve("b") as job, gate.run(job):
            pass
    release.set()
    await t
    # The slot must be usable again after a timed-out wait.
    async with gate.reserve("c") as job, gate.run(job):
        assert gate.active_jobs == 1


async def test_long_running_job_rejects_at_reserve():
    """A job estimated to outlast the wait budget is rejected before download."""
    gate = JobGate(
        max_concurrent=1, max_queued=1, queue_wait_seconds=15, speed_factor=8
    )
    release = asyncio.Event()

    async def hold():
        # 1 hour of audio at 8x => ~450s remaining, far beyond 15s.
        async with gate.reserve("a") as job, gate.run(job, audio_seconds=3600):
            await release.wait()

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    with pytest.raises(ServiceBusyError) as exc:
        async with gate.reserve("b"):
            pass
    assert 400 <= exc.value.retry_after <= 460
    snap = gate.snapshot()
    assert snap["busy"] is True
    assert snap["active_jobs"] == 1
    assert snap["estimated_wait_seconds"] >= 400
    release.set()
    await t


async def test_short_running_job_allows_queueing():
    gate = JobGate(
        max_concurrent=1, max_queued=1, queue_wait_seconds=15, speed_factor=8
    )
    release = asyncio.Event()

    async def hold():
        # 20s of audio => ~4.5s estimated, within the 15s budget.
        async with gate.reserve("a") as job, gate.run(job, audio_seconds=20):
            await release.wait()

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    async with gate.reserve("b"):
        assert gate.queued_jobs == 1
    release.set()
    await t


def test_snapshot_idle():
    gate = JobGate()
    assert gate.snapshot() == {
        "busy": False,
        "active_jobs": 0,
        "queued_jobs": 0,
        "max_concurrent_jobs": 1,
        "max_queued_jobs": 1,
        "estimated_wait_seconds": None,
    }
    assert gate.retry_after() == 5


@pytest.mark.parametrize("kwargs", [{"max_concurrent": 0}, {"max_queued": -1}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        JobGate(**kwargs)


async def test_run_rejects_when_long_job_started_after_reserve():
    """reserve() passed while idle, but a long job grabbed the GPU before run()."""
    gate = JobGate(
        max_concurrent=1, max_queued=1, queue_wait_seconds=15, speed_factor=8
    )
    release = asyncio.Event()

    async with gate.reserve("b") as job_b:  # admitted: nothing running yet

        async def hold():
            async with gate.reserve("a") as job_a:
                async with gate.run(job_a, audio_seconds=3600):
                    await release.wait()

        t = asyncio.create_task(hold())
        await asyncio.sleep(0.01)
        with pytest.raises(ServiceBusyError) as exc:
            async with gate.run(job_b, audio_seconds=1):
                pass
        assert exc.value.retry_after >= 400
        release.set()
        await t
