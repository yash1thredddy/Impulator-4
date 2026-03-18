"""
Test FOUND-05 (contextvars propagation to ThreadPoolExecutor workers).

Validates that:
- request_id set in calling thread is visible in worker thread
- Empty/default request_id propagates correctly
- Different requests get isolated context snapshots (copy_context semantics)
"""
import threading

from backend.core.logging import request_id_var, session_id_var
from backend.core.executor import JobExecutor


class TestExecutorContextPropagation:
    """Verify contextvars propagation through JobExecutor.submit()."""

    def test_executor_propagates_request_id_to_worker(self):
        """request_id set before submit() is visible inside the worker thread."""
        captured = {}
        done_event = threading.Event()

        token_rid = request_id_var.set("test-request-id-abc")
        token_sid = session_id_var.set("test-session-id-xyz")

        executor = JobExecutor(max_workers=1)
        try:
            def worker(job_id):
                captured["request_id"] = request_id_var.get("")
                captured["session_id"] = session_id_var.get("")
                captured["thread"] = threading.current_thread().name
                done_event.set()

            executor.submit("test-job-1", worker)
            assert done_event.wait(timeout=5), "Worker did not complete within 5s"

            assert captured["request_id"] == "test-request-id-abc", (
                f"Expected 'test-request-id-abc', got: {captured['request_id']}"
            )
            assert captured["session_id"] == "test-session-id-xyz", (
                f"Expected 'test-session-id-xyz', got: {captured['session_id']}"
            )
            # Verify it actually ran in a different thread
            assert "job_worker" in captured["thread"], (
                f"Expected worker thread, got: {captured['thread']}"
            )
        finally:
            executor.shutdown(wait=True)
            request_id_var.reset(token_rid)
            session_id_var.reset(token_sid)

    def test_executor_without_context_gets_default(self):
        """Worker gets empty string (default) when no request_id is set."""
        captured = {}
        done_event = threading.Event()

        # Ensure default value
        token = request_id_var.set("")

        executor = JobExecutor(max_workers=1)
        try:
            def worker(job_id):
                captured["request_id"] = request_id_var.get("")
                done_event.set()

            executor.submit("test-job-default", worker)
            assert done_event.wait(timeout=5), "Worker did not complete within 5s"

            assert captured["request_id"] == "", (
                f"Expected empty default, got: {captured['request_id']}"
            )
        finally:
            executor.shutdown(wait=True)
            request_id_var.reset(token)

    def test_executor_different_requests_get_different_context(self):
        """copy_context() takes a snapshot -- two jobs get isolated copies."""
        captured_1 = {}
        captured_2 = {}
        event_1 = threading.Event()
        event_2 = threading.Event()
        # Gate so both workers run and capture before either finishes cleanup
        gate = threading.Event()

        token2 = None  # Initialize before try so finally can safely check
        executor = JobExecutor(max_workers=2)
        try:
            def worker_1(job_id):
                captured_1["request_id"] = request_id_var.get("")
                event_1.set()
                gate.wait(timeout=5)

            def worker_2(job_id):
                captured_2["request_id"] = request_id_var.get("")
                event_2.set()
                gate.wait(timeout=5)

            # Submit job 1 with rid-1
            token1 = request_id_var.set("rid-1")
            executor.submit("test-iso-1", worker_1)

            # Submit job 2 with rid-2
            token2 = request_id_var.set("rid-2")
            executor.submit("test-iso-2", worker_2)

            # Wait for both workers to capture their values
            assert event_1.wait(timeout=5), "Worker 1 did not start within 5s"
            assert event_2.wait(timeout=5), "Worker 2 did not start within 5s"

            # Release gate so workers finish
            gate.set()

            assert captured_1["request_id"] == "rid-1", (
                f"Job 1 expected 'rid-1', got: {captured_1['request_id']}"
            )
            assert captured_2["request_id"] == "rid-2", (
                f"Job 2 expected 'rid-2', got: {captured_2['request_id']}"
            )
        finally:
            gate.set()  # Ensure workers unblock on failure path
            executor.shutdown(wait=True)
            if token2 is not None:
                request_id_var.reset(token2)
            request_id_var.reset(token1)
