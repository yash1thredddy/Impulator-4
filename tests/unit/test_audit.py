"""
Unit tests for Audit Logging Module.

Tests security event logging including:
- Event types and structure
- Rate limit logging
- Job operation logging
- Validation failure logging
- Security threat logging
"""
import pytest
from unittest.mock import patch, MagicMock
import json


class TestAuditEvent:
    """Tests for AuditEvent enumeration."""

    def test_audit_event_values(self):
        """Test that all audit event types have correct values."""
        from backend.core.audit import AuditEvent

        assert AuditEvent.RATE_LIMIT_EXCEEDED.value == "rate_limit_exceeded"
        assert AuditEvent.JOB_CANCELLED.value == "job_cancelled"
        assert AuditEvent.JOB_DELETED.value == "job_deleted"
        assert AuditEvent.BATCH_CANCELLED.value == "batch_cancelled"
        assert AuditEvent.VALIDATION_FAILED.value == "validation_failed"
        assert AuditEvent.PATH_TRAVERSAL_BLOCKED.value == "path_traversal_blocked"
        assert AuditEvent.INVALID_SMILES.value == "invalid_smiles"
        assert AuditEvent.AUTHENTICATION_FAILED.value == "authentication_failed"
        assert AuditEvent.AUTHORIZATION_FAILED.value == "authorization_failed"
        assert AuditEvent.SUSPICIOUS_INPUT.value == "suspicious_input"


class TestLogSecurityEvent:
    """Tests for log_security_event function."""

    @patch('backend.core.audit.audit_logger')
    def test_log_event_info_severity(self, mock_logger):
        """Test logging event with info severity."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.JOB_CANCELLED,
            session_id="test-session",
            details={"job_id": "123"},
            severity="info"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "job_cancelled"
        assert log_data["session_id"] == "test-session"
        assert log_data["details"]["job_id"] == "123"
        assert "timestamp" in log_data

    @patch('backend.core.audit.audit_logger')
    def test_log_event_warning_severity(self, mock_logger):
        """Test logging event with warning severity."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.RATE_LIMIT_EXCEEDED,
            session_id="test-session",
            severity="warning"
        )

        mock_logger.warning.assert_called_once()

    @patch('backend.core.audit.audit_logger')
    def test_log_event_error_severity(self, mock_logger):
        """Test logging event with error severity."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.PATH_TRAVERSAL_BLOCKED,
            details={"path": "/etc/passwd"},
            severity="error"
        )

        mock_logger.error.assert_called_once()

    @patch('backend.core.audit.audit_logger')
    def test_log_event_critical_severity(self, mock_logger):
        """Test logging event with critical severity."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.AUTHENTICATION_FAILED,
            severity="critical"
        )

        mock_logger.critical.assert_called_once()

    @patch('backend.core.audit.audit_logger')
    def test_log_event_anonymous_session(self, mock_logger):
        """Test logging event with no session ID."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.SUSPICIOUS_INPUT,
            session_id=None
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["session_id"] == "anonymous"

    @patch('backend.core.audit.audit_logger')
    def test_log_event_empty_details(self, mock_logger):
        """Test logging event with no details."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.JOB_DELETED,
            details=None
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["details"] == {}


class TestLogRateLimitExceeded:
    """Tests for log_rate_limit_exceeded function."""

    @patch('backend.core.audit.audit_logger')
    def test_rate_limit_logging(self, mock_logger):
        """Test rate limit exceeded logging."""
        from backend.core.audit import log_rate_limit_exceeded

        log_rate_limit_exceeded(
            session_id="user-123",
            limit_type="jobs_per_hour",
            limit_value=10
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "rate_limit_exceeded"
        assert log_data["session_id"] == "user-123"
        assert log_data["details"]["limit_type"] == "jobs_per_hour"
        assert log_data["details"]["limit_value"] == 10


class TestLogJobCancelled:
    """Tests for log_job_cancelled function."""

    @patch('backend.core.audit.audit_logger')
    def test_job_cancelled_logging(self, mock_logger):
        """Test job cancelled logging."""
        from backend.core.audit import log_job_cancelled

        log_job_cancelled(
            session_id="user-123",
            job_id="job-456",
            compound_name="Aspirin"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "job_cancelled"
        assert log_data["details"]["job_id"] == "job-456"
        assert log_data["details"]["compound_name"] == "Aspirin"

    @patch('backend.core.audit.audit_logger')
    def test_job_cancelled_without_name(self, mock_logger):
        """Test job cancelled logging without compound name."""
        from backend.core.audit import log_job_cancelled

        log_job_cancelled(
            session_id="user-123",
            job_id="job-456"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["details"]["compound_name"] is None


class TestLogJobDeleted:
    """Tests for log_job_deleted function."""

    @patch('backend.core.audit.audit_logger')
    def test_job_deleted_logging(self, mock_logger):
        """Test job deleted logging."""
        from backend.core.audit import log_job_deleted

        log_job_deleted(
            session_id="user-123",
            job_id="job-456",
            compound_name="Caffeine"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "job_deleted"
        assert log_data["details"]["job_id"] == "job-456"


class TestLogValidationFailed:
    """Tests for log_validation_failed function."""

    @patch('backend.core.audit.audit_logger')
    def test_validation_failed_logging(self, mock_logger):
        """Test validation failed logging."""
        from backend.core.audit import log_validation_failed

        log_validation_failed(
            session_id="user-123",
            field="smiles",
            value="invalid_smiles",
            reason="Invalid SMILES syntax"
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "validation_failed"
        assert log_data["details"]["field"] == "smiles"
        assert log_data["details"]["reason"] == "Invalid SMILES syntax"

    @patch('backend.core.audit.audit_logger')
    def test_validation_failed_truncates_long_value(self, mock_logger):
        """Test validation failed truncates long values."""
        from backend.core.audit import log_validation_failed

        long_value = "A" * 200

        log_validation_failed(
            session_id="user-123",
            field="compound_name",
            value=long_value,
            reason="Name too long"
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        # Value should be truncated to 100 chars + "..."
        assert len(log_data["details"]["value"]) == 103
        assert log_data["details"]["value"].endswith("...")


class TestLogPathTraversalBlocked:
    """Tests for log_path_traversal_blocked function."""

    @patch('backend.core.audit.audit_logger')
    def test_path_traversal_logging(self, mock_logger):
        """Test path traversal blocked logging."""
        from backend.core.audit import log_path_traversal_blocked

        log_path_traversal_blocked(
            attempted_path="../../../etc/passwd"
        )

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "path_traversal_blocked"
        assert "../../../etc/passwd" in log_data["details"]["attempted_path"]


class TestLogSuspiciousInput:
    """Tests for log_suspicious_input function."""

    @patch('backend.core.audit.audit_logger')
    def test_suspicious_input_logging(self, mock_logger):
        """Test suspicious input logging."""
        from backend.core.audit import log_suspicious_input

        log_suspicious_input(
            session_id="user-123",
            field="compound_name",
            pattern_matched="sql_injection"
        )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        log_data = json.loads(call_args)

        assert log_data["event"] == "suspicious_input"
        assert log_data["details"]["field"] == "compound_name"
        assert log_data["details"]["pattern_matched"] == "sql_injection"


class TestAuditLogStructure:
    """Tests for audit log entry structure."""

    @patch('backend.core.audit.audit_logger')
    def test_log_entry_has_timestamp(self, mock_logger):
        """Test that all log entries have ISO timestamp."""
        from backend.core.audit import log_security_event, AuditEvent
        import re

        # Use severity="info" to call audit_logger.info
        log_security_event(AuditEvent.JOB_CANCELLED, session_id="test", severity="info")

        call_args = mock_logger.info.call_args[0][0]
        log_data = json.loads(call_args)

        # Check ISO 8601 format
        iso_pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
        assert re.match(iso_pattern, log_data["timestamp"])

    @patch('backend.core.audit.audit_logger')
    def test_log_entry_is_valid_json(self, mock_logger):
        """Test that all log entries are valid JSON."""
        from backend.core.audit import log_security_event, AuditEvent

        log_security_event(
            AuditEvent.VALIDATION_FAILED,
            session_id="test",
            details={"key": "value", "nested": {"a": 1}}
        )

        call_args = mock_logger.warning.call_args[0][0]

        # Should not raise
        parsed = json.loads(call_args)
        assert isinstance(parsed, dict)
