"""
Structured logging configuration and correlation ID middleware.

Provides:
- request_id_var / session_id_var: ContextVar declarations for correlation IDs
- configure_logging(): structlog + stdlib dictConfig setup
- CorrelationIdMiddleware: FastAPI middleware for request_id injection
- HealthProbeFilter: Downgrades health probe logs to DEBUG level
"""
import contextvars
import logging
import logging.config
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Context variables -- accessible from any thread via copy_context()
# ---------------------------------------------------------------------------

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default=""
)


# ---------------------------------------------------------------------------
# Health probe log filter
# ---------------------------------------------------------------------------


class HealthProbeFilter(logging.Filter):
    """Downgrade health probe access logs to DEBUG level.

    Prevents /health/live and /health/ready from flooding INFO logs
    while still keeping them visible at DEBUG level.
    """

    PROBE_PATHS = {"/api/v1/health/live", "/api/v1/health/ready"}

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for path in self.PROBE_PATHS:
            if path in msg:
                record.levelno = logging.DEBUG
                record.levelname = "DEBUG"
        return True


# ---------------------------------------------------------------------------
# structlog + stdlib logging configuration
# ---------------------------------------------------------------------------


def configure_logging() -> None:
    """Configure structlog to wrap stdlib logging.

    Call once at app startup, BEFORE any logging calls.
    Reads LOG_LEVEL and LOG_FORMAT from config.py (env vars).

    After this call:
    - Existing ``logging.getLogger(__name__)`` calls produce structured output
    - New code can use ``structlog.get_logger()`` for key-value style logging
    - All log entries include request_id and session_id from contextvars
    """
    from backend.config import settings

    # Determine output format
    use_json = settings.LOG_FORMAT == "json" or (
        settings.LOG_FORMAT == "auto" and not settings.DEBUG
    )
    log_level = settings.LOG_LEVEL

    # Shared processors for ALL log entries (structlog + stdlib foreign)
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Choose renderer based on environment
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog itself
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging via dictConfig
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        renderer,
                    ],
                    "foreign_pre_chain": shared_processors,
                },
            },
            "filters": {
                "health_probe": {
                    "()": HealthProbeFilter,
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structlog",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "()": "backend.core.azure_sync.AzureSyncRotatingFileHandler",
                    "filename": "data/logs/backend.log",
                    "maxBytes": 10_000_000,
                    "backupCount": 2,
                    "encoding": "utf-8",
                    "formatter": "structlog",
                },
                "audit_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "data/logs/audit.log",
                    "maxBytes": 10_485_760,
                    "backupCount": 5,
                    "encoding": "utf-8",
                    "formatter": "structlog",
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console", "file"],
                    "level": log_level,
                },
                "audit": {
                    "handlers": ["console", "file", "audit_file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["health_probe"],
                },
            },
        }
    )


# ---------------------------------------------------------------------------
# Correlation ID middleware
# ---------------------------------------------------------------------------


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generate a unique request_id per HTTP request.

    - Stores request_id and session_id in contextvars
    - Binds them to structlog context (auto-injected into all log lines)
    - Adds X-Request-ID response header
    """

    async def dispatch(self, request: Request, call_next):
        # Generate correlation ID (server-generated, ignore client-sent)
        rid = str(uuid.uuid4())
        request_id_var.set(rid)

        # Extract session_id from header if present
        sid = request.headers.get("X-Session-ID", "")
        session_id_var.set(sid)

        # Bind to structlog context (auto-injected into all log lines)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=rid,
            session_id=sid,
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response
