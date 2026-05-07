"""
Standard error handling: ErrorCode enum, AppException, and FastAPI exception handlers.

Provides:
- ErrorCode StrEnum with ~14 machine-readable error codes
- AppException for raising errors with explicit error_code
- STATUS_TO_ERROR_CODE mapping for inferring error_code from HTTP status
- Three async exception handlers for FastAPI registration
"""
from enum import StrEnum

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.logging import request_id_var


class ErrorCode(StrEnum):
    """Machine-readable error codes for API responses.

    Fixed set of 14 codes at medium granularity.
    Frontend can switch on these reliably; `detail` provides human-readable specifics.
    Documented in OpenAPI schema via StrEnum.
    """

    # Core codes
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    DUPLICATE_FOUND = "DUPLICATE_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    DOWNSTREAM_FAILURE = "DOWNSTREAM_FAILURE"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # Validation subtypes
    SMILES_INVALID = "SMILES_INVALID"
    BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
    MISSING_FIELD = "MISSING_FIELD"

    # Downstream subtypes
    CHEMBL_FAILURE = "CHEMBL_FAILURE"
    PDB_FAILURE = "PDB_FAILURE"
    CLASSYFIRE_FAILURE = "CLASSYFIRE_FAILURE"

    # Job lifecycle
    JOB_TIMEOUT = "JOB_TIMEOUT"
    SYNC_FAILED = "SYNC_FAILED"


# Default mapping: HTTP status code -> ErrorCode
# Used by http_exception_handler when no explicit error_code is provided.
STATUS_TO_ERROR_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.VALIDATION_ERROR,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    422: ErrorCode.VALIDATION_ERROR,
    429: ErrorCode.RATE_LIMITED,
    500: ErrorCode.INTERNAL_ERROR,
    503: ErrorCode.DOWNSTREAM_FAILURE,
}


class AppException(Exception):
    """Application exception with explicit error_code for standard error response.

    Use for new code that wants to set error_code explicitly rather than
    relying on STATUS_TO_ERROR_CODE inference from HTTP status.
    """

    def __init__(self, status_code: int, detail: str, error_code: ErrorCode):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)


# ---------------------------------------------------------------------------
# FastAPI exception handlers (register in main.py via app.exception_handler)
# ---------------------------------------------------------------------------


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> ORJSONResponse:
    """Handle StarletteHTTPException (and FastAPI HTTPException which inherits it).

    Infers error_code from status code via STATUS_TO_ERROR_CODE.
    If exc.detail is a dict with an explicit 'error_code' key, uses that instead.
    """
    rid = request_id_var.get("")
    error_code = STATUS_TO_ERROR_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)

    # Allow explicit error_code passed via detail dict
    # Copy before mutating — exc.detail is the caller's original dict
    detail = exc.detail
    if isinstance(detail, dict) and "error_code" in detail:
        detail = dict(detail)  # shallow copy to avoid mutating the exception
        error_code = detail.pop("error_code")
        detail = detail.get("detail", str(detail))

    return ORJSONResponse(
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
        content={
            "detail": detail if isinstance(detail, str) else str(detail),
            "error_code": str(error_code),
            "request_id": rid,
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> ORJSONResponse:
    """Handle Pydantic RequestValidationError (422 responses).

    Returns standard error shape with additional 'errors' array
    containing field-level validation details.
    """
    rid = request_id_var.get("")
    return ORJSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "error_code": str(ErrorCode.VALIDATION_ERROR),
            "request_id": rid,
            "errors": [
                {
                    "field": ".".join(str(loc) for loc in err["loc"]),
                    "message": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ],
        },
    )


async def app_exception_handler(
    request: Request, exc: "AppException"
) -> ORJSONResponse:
    """Handle AppException with explicit error_code."""
    rid = request_id_var.get("")
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": str(exc.error_code),
            "request_id": rid,
        },
    )
