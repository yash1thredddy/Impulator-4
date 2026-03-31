"""
Test FOUND-01 (error response shape) and FOUND-02 (HTTP status codes).

Validates that all error responses conform to the standard shape:
    {detail: str, error_code: str, request_id: str}
across 404, 422, and 500 status codes, and that error_code values
match the STATUS_TO_ERROR_CODE mapping.
"""
import re
from backend.core.exceptions import ErrorCode


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class TestErrorResponseShape:
    """Verify every error response has {detail, error_code, request_id}."""

    def test_404_returns_standard_error_shape(self, client):
        """GET a nonexistent compound returns 404 with standard shape."""
        resp = client.get("/api/v1/compounds/00000000-0000-4000-8000-000000000007")
        assert resp.status_code == 404

        body = resp.json()
        assert "detail" in body
        assert "error_code" in body
        assert "request_id" in body
        assert body["error_code"] == str(ErrorCode.NOT_FOUND)
        assert UUID_PATTERN.match(body["request_id"]), (
            f"request_id should be UUID, got: {body['request_id']}"
        )

    def test_422_validation_error_shape(self, client):
        """POST /jobs with empty body returns 422 with errors array."""
        resp = client.post("/api/v1/jobs", json={})
        assert resp.status_code == 422

        body = resp.json()
        # Standard keys
        assert "detail" in body
        assert "error_code" in body
        assert "request_id" in body
        assert body["error_code"] == str(ErrorCode.VALIDATION_ERROR)

        # Validation-specific: errors array
        assert "errors" in body
        assert isinstance(body["errors"], list)
        assert len(body["errors"]) >= 1

        # Each error entry has field, message, type
        for err in body["errors"]:
            assert "field" in err, f"Missing 'field' in error entry: {err}"
            assert "message" in err, f"Missing 'message' in error entry: {err}"
            assert "type" in err, f"Missing 'type' in error entry: {err}"

    def test_500_internal_error_shape(self, pg_engine, mock_azure):
        """Unhandled exception produces standard error shape with INTERNAL_ERROR."""
        from backend.main import app
        from backend.core import database as db_module
        from backend.core.database import get_db
        from sqlalchemy.orm import sessionmaker
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        # Need a TestClient with raise_server_exceptions=False to inspect 500 body
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=pg_engine)
        original_engine = db_module.engine
        original_session_local = db_module.SessionLocal
        db_module.engine = pg_engine
        db_module.SessionLocal = TestSessionLocal

        def override_get_db():
            session = TestSessionLocal()
            try:
                yield session
            finally:
                session.close()

        app.dependency_overrides[get_db] = override_get_db

        # Add a temporary route that raises an unhandled exception
        async def raise_unhandled():
            raise RuntimeError("deliberate test explosion")

        app.add_api_route("/api/v1/_test_500", raise_unhandled, methods=["GET"])
        try:
            with patch('backend.core.scheduler.trigger'):
                with TestClient(app, raise_server_exceptions=False) as c:
                    resp = c.get("/api/v1/_test_500")
                    assert resp.status_code == 500

                    body = resp.json()
                    assert "detail" in body
                    assert "error_code" in body
                    assert "request_id" in body
                    assert body["error_code"] == str(ErrorCode.INTERNAL_ERROR)
        finally:
            # Remove the temporary route and restore overrides
            app.routes[:] = [
                r for r in app.routes
                if getattr(r, "path", "") != "/api/v1/_test_500"
            ]
            app.dependency_overrides.pop(get_db, None)
            # Restore the original db_module state so other tests are not affected
            db_module.engine = original_engine
            db_module.SessionLocal = original_session_local

    def test_error_response_always_has_request_id(self, client):
        """Multiple different error types all include unique request_ids."""
        request_ids = []

        # 404
        r1 = client.get("/api/v1/compounds/00000000-0000-4000-8000-00000000000a")
        assert r1.status_code == 404
        request_ids.append(r1.json()["request_id"])

        # 422
        r2 = client.post("/api/v1/jobs", json={})
        assert r2.status_code == 422
        request_ids.append(r2.json()["request_id"])

        # Another 404
        r3 = client.get("/api/v1/compounds/00000000-0000-4000-8000-00000000000b")
        assert r3.status_code == 404
        request_ids.append(r3.json()["request_id"])

        # All present
        for rid in request_ids:
            assert rid, "request_id should be non-empty"
            assert UUID_PATTERN.match(rid), f"request_id should be UUID, got: {rid}"

        # All unique
        assert len(set(request_ids)) == len(request_ids), (
            f"request_ids should be unique, got: {request_ids}"
        )

    def test_error_code_matches_status(self, client):
        """Verify error_code corresponds to the HTTP status code."""
        # 404 -> NOT_FOUND
        r404 = client.get("/api/v1/compounds/00000000-0000-4000-8000-000000000009")
        assert r404.status_code == 404
        assert r404.json()["error_code"] == "NOT_FOUND"

        # 422 -> VALIDATION_ERROR
        r422 = client.post("/api/v1/jobs", json={})
        assert r422.status_code == 422
        assert r422.json()["error_code"] == "VALIDATION_ERROR"
