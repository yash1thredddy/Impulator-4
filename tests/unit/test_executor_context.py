"""
Test contextvars propagation to asyncio.Task (D-12, D-80).

asyncio copies context at task creation automatically -- no explicit
copy_context() needed (unlike ThreadPoolExecutor).

asyncio_mode = auto in pytest.ini -- no @pytest.mark.asyncio needed.
"""
import asyncio

from backend.core.logging import request_id_var, session_id_var
from backend.core import executor


class TestAsyncContextPropagation:
    """Verify contextvars propagation through asyncio.Task."""

    def setup_method(self):
        executor._reset()

    def teardown_method(self):
        executor._reset()

    async def test_contextvars_propagate_to_task(self):
        """request_id set before submit() is visible inside the async task."""
        captured = {}
        done = asyncio.Event()

        token_rid = request_id_var.set("test-request-id-abc")
        token_sid = session_id_var.set("test-session-id-xyz")

        try:
            async def worker(job_id, **kwargs):
                captured["request_id"] = request_id_var.get("")
                captured["session_id"] = session_id_var.get("")
                done.set()

            await executor.submit("ctx-test-1", worker)
            await asyncio.wait_for(done.wait(), timeout=5)

            assert captured["request_id"] == "test-request-id-abc"
            assert captured["session_id"] == "test-session-id-xyz"
        finally:
            request_id_var.reset(token_rid)
            session_id_var.reset(token_sid)

    async def test_default_context_when_not_set(self):
        """Task gets default empty string when no contextvar is set."""
        captured = {}
        done = asyncio.Event()

        token = request_id_var.set("")

        try:
            async def worker(job_id, **kwargs):
                captured["request_id"] = request_id_var.get("")
                done.set()

            await executor.submit("ctx-default", worker)
            await asyncio.wait_for(done.wait(), timeout=5)

            assert captured["request_id"] == ""
        finally:
            request_id_var.reset(token)

    async def test_isolated_context_between_tasks(self):
        """Two tasks get isolated context snapshots."""
        captured_1 = {}
        captured_2 = {}
        event_1 = asyncio.Event()
        event_2 = asyncio.Event()
        gate = asyncio.Event()

        async def worker_1(job_id, **kwargs):
            captured_1["request_id"] = request_id_var.get("")
            event_1.set()
            await gate.wait()

        async def worker_2(job_id, **kwargs):
            captured_2["request_id"] = request_id_var.get("")
            event_2.set()
            await gate.wait()

        # Submit task 1 with rid-1
        token1 = request_id_var.set("rid-1")
        await executor.submit("iso-1", worker_1)

        # Submit task 2 with rid-2
        token2 = request_id_var.set("rid-2")
        await executor.submit("iso-2", worker_2)

        try:
            await asyncio.wait_for(event_1.wait(), timeout=5)
            await asyncio.wait_for(event_2.wait(), timeout=5)

            gate.set()

            assert captured_1["request_id"] == "rid-1"
            assert captured_2["request_id"] == "rid-2"
        finally:
            gate.set()
            request_id_var.reset(token2)
            request_id_var.reset(token1)
            await asyncio.sleep(0.05)
