"""
Unit tests for authentication and session validation.
"""
import pytest
from unittest.mock import MagicMock

# Skip all tests if fastapi not installed
pytest.importorskip("fastapi")
from fastapi import HTTPException


class TestSessionValidation:
    """Tests for session ID validation."""

    def test_valid_uuid_session_id(self):
        """Test that valid UUID session IDs are accepted."""
        from backend.core.auth import validate_session_id

        # Valid UUID v4
        valid_id = "550e8400-e29b-41d4-a716-446655440000"
        result = validate_session_id(valid_id)
        assert result == valid_id

    def test_valid_uuid_uppercase(self):
        """Test that uppercase UUIDs are accepted."""
        from backend.core.auth import validate_session_id

        valid_id = "550E8400-E29B-41D4-A716-446655440000"
        result = validate_session_id(valid_id)
        assert result == valid_id

    def test_none_session_generates_anonymous(self):
        """Test that None session ID generates anonymous session."""
        from backend.core.auth import validate_session_id

        result = validate_session_id(None)
        assert result.startswith("anon-")
        # Verify the generated part is a valid UUID
        anon_uuid = result[5:]  # Remove 'anon-' prefix
        assert len(anon_uuid) == 36

    def test_invalid_format_raises_error(self):
        """Test that invalid session ID format raises HTTPException."""
        from backend.core.auth import validate_session_id

        with pytest.raises(HTTPException) as exc_info:
            validate_session_id("invalid-session-id")

        assert exc_info.value.status_code == 400
        assert "Invalid session ID format" in exc_info.value.detail

    def test_injection_attempt_rejected(self):
        """Test that SQL injection attempts are rejected."""
        from backend.core.auth import validate_session_id

        malicious_ids = [
            "'; DROP TABLE jobs; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "550e8400-e29b-41d4-a716-446655440000; DELETE FROM",
        ]

        for malicious_id in malicious_ids:
            with pytest.raises(HTTPException) as exc_info:
                validate_session_id(malicious_id)
            assert exc_info.value.status_code == 400

    def test_whitespace_stripped(self):
        """Test that whitespace is stripped from session ID."""
        from backend.core.auth import validate_session_id

        valid_id = "550e8400-e29b-41d4-a716-446655440000"
        result = validate_session_id(f"  {valid_id}  ")
        assert result == valid_id

    def test_empty_string_generates_anonymous(self):
        """Test that empty string session ID generates anonymous session."""
        from backend.core.auth import validate_session_id

        # Empty string after strip becomes falsy
        result = validate_session_id("")
        assert result.startswith("anon-")

    def test_non_uuid_v4_rejected(self):
        """Test that non-v4 UUIDs are rejected."""
        from backend.core.auth import validate_session_id

        # UUID v1 (first digit after second hyphen is not 4)
        uuid_v1 = "550e8400-e29b-11d4-a716-446655440000"
        with pytest.raises(HTTPException):
            validate_session_id(uuid_v1)


class TestTruncateSessionId:
    """Tests for session ID truncation in logs."""

    def test_truncate_valid_session(self):
        """Test truncation of valid session ID."""
        from backend.core.auth import truncate_session_id

        session_id = "550e8400-e29b-41d4-a716-446655440000"
        result = truncate_session_id(session_id)
        assert result == "550e8400..."

    def test_truncate_none_returns_unknown(self):
        """Test truncation of None returns 'unknown'."""
        from backend.core.auth import truncate_session_id

        result = truncate_session_id(None)
        assert result == "unknown"

    def test_truncate_empty_returns_unknown(self):
        """Test truncation of empty string returns 'unknown'."""
        from backend.core.auth import truncate_session_id

        result = truncate_session_id("")
        assert result == "unknown"

    def test_truncate_short_session(self):
        """Test truncation of short session ID."""
        from backend.core.auth import truncate_session_id

        short_id = "abc"
        result = truncate_session_id(short_id)
        assert result == "abc..."
