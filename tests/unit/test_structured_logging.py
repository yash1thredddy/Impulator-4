"""
Test FOUND-04 (structured JSON logging with required fields).

Validates that:
- configure_logging() produces JSON output when LOG_FORMAT="json"
- JSON output contains level, logger_name, timestamp, event keys
- Console mode works without errors
- Auto mode selects json/console based on DEBUG flag
- Contextvars (request_id, session_id) are injected into log output
"""
import json
import logging
import io

import structlog
from unittest.mock import patch


def _capture_log_output(logger_name: str, message: str) -> str:
    """Configure logging, capture output from a stdlib logger, return raw string."""
    from backend.core.logging import configure_logging

    configure_logging()

    # Attach a StringIO handler to capture output
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)

    # Use the same formatter structlog configured on the console handler
    root = logging.getLogger()
    for h in root.handlers:
        if hasattr(h, "formatter") and h.formatter is not None:
            handler.setFormatter(h.formatter)
            break

    test_logger = logging.getLogger(logger_name)
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)
    try:
        test_logger.info(message)
        handler.flush()
        return stream.getvalue()
    finally:
        test_logger.removeHandler(handler)


class TestStructuredLogging:
    """Verify structured logging configuration and output format."""

    def test_configure_logging_produces_json_output(self):
        """JSON format produces parseable JSON with required keys."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.LOG_FORMAT = "json"
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.DEBUG = False

            output = _capture_log_output("test_json_module", "test json message")

        # Should be valid JSON
        data = json.loads(output.strip())
        assert "level" in data, f"Missing 'level' key in: {data}"
        assert "logger" in data or "logger_name" in data, (
            f"Missing logger identifier in: {data}"
        )
        assert "timestamp" in data, f"Missing 'timestamp' key in: {data}"
        assert "event" in data, f"Missing 'event' key in: {data}"
        assert data["event"] == "test json message"

    def test_configure_logging_console_mode(self):
        """Console format configures without errors."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.LOG_FORMAT = "console"
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.DEBUG = True

            from backend.core.logging import configure_logging
            # Should not raise
            configure_logging()

    def test_log_format_auto_uses_json_in_production(self):
        """auto format selects JSON when DEBUG=False."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.LOG_FORMAT = "auto"
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.DEBUG = False

            output = _capture_log_output("test_auto_prod", "auto prod message")

        # Should be valid JSON
        data = json.loads(output.strip())
        assert "event" in data
        assert data["event"] == "auto prod message"

    def test_log_format_auto_uses_console_in_debug(self):
        """auto format selects console (non-JSON) when DEBUG=True."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.LOG_FORMAT = "auto"
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.DEBUG = True

            output = _capture_log_output("test_auto_debug", "auto debug message")

        # Console output should NOT be valid JSON
        try:
            json.loads(output.strip())
            is_json = True
        except (json.JSONDecodeError, ValueError):
            is_json = False

        assert not is_json, f"Expected non-JSON console output, got: {output}"

    def test_contextvars_injected_into_log_output(self):
        """request_id and session_id from contextvars appear in JSON log output."""
        with patch("backend.config.settings") as mock_settings:
            mock_settings.LOG_FORMAT = "json"
            mock_settings.LOG_LEVEL = "INFO"
            mock_settings.DEBUG = False

            from backend.core.logging import configure_logging
            configure_logging()

            # Bind contextvars
            structlog.contextvars.bind_contextvars(
                request_id="test-rid-123",
                session_id="test-sid-456",
            )
            try:
                output = _capture_log_output("test_ctx_module", "context test")
                data = json.loads(output.strip())

                assert data.get("request_id") == "test-rid-123", (
                    f"Expected request_id='test-rid-123', got: {data}"
                )
                assert data.get("session_id") == "test-sid-456", (
                    f"Expected session_id='test-sid-456', got: {data}"
                )
            finally:
                structlog.contextvars.clear_contextvars()
