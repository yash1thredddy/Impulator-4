"""
Test FOUND-03 (correlation ID header on every response).

Validates that:
- Every HTTP response includes an X-Request-ID header
- Header value is a valid UUID
- Error response body request_id matches the header
- Each request gets a unique ID (server-generated, client value ignored)
"""
import re
import pytest


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class TestCorrelationIdHeader:
    """Verify X-Request-ID header presence, format, and uniqueness."""

    def test_success_response_has_x_request_id_header(self, client):
        """Successful response includes X-Request-ID header with UUID value."""
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200

        rid = resp.headers.get("X-Request-ID")
        assert rid is not None, "X-Request-ID header missing from success response"
        assert UUID_PATTERN.match(rid), f"X-Request-ID should be UUID, got: {rid}"

    def test_error_response_has_x_request_id_header(self, client):
        """Error response includes X-Request-ID header matching body request_id."""
        resp = client.get("/api/v1/compounds/nonexistent-id")
        assert resp.status_code == 404

        header_rid = resp.headers.get("X-Request-ID")
        assert header_rid is not None, "X-Request-ID header missing from error response"

        body_rid = resp.json()["request_id"]
        assert header_rid == body_rid, (
            f"Header X-Request-ID ({header_rid}) should match body request_id ({body_rid})"
        )

    def test_each_request_gets_unique_request_id(self, client):
        """Five consecutive requests each get a different X-Request-ID."""
        ids = set()
        for _ in range(5):
            resp = client.get("/api/v1/health/live")
            assert resp.status_code == 200
            rid = resp.headers.get("X-Request-ID")
            assert rid is not None
            ids.add(rid)

        assert len(ids) == 5, f"Expected 5 unique IDs, got {len(ids)}: {ids}"

    def test_client_sent_x_request_id_is_ignored(self, client):
        """Server generates its own X-Request-ID, ignoring client-sent value."""
        client_rid = "client-sent-id-should-be-ignored"
        resp = client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": client_rid},
        )
        assert resp.status_code == 200

        server_rid = resp.headers.get("X-Request-ID")
        assert server_rid is not None
        assert server_rid != client_rid, (
            "Server should generate its own X-Request-ID, not echo client value"
        )
        assert UUID_PATTERN.match(server_rid), (
            f"Server X-Request-ID should be UUID, got: {server_rid}"
        )
