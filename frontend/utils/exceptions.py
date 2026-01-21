"""Custom exceptions for IMPULATOR frontend.

This module defines a hierarchy of exceptions for better error handling
and more informative error messages.

Exception Hierarchy:
    ImpulatorError (base)
    ├── InvalidSMILESError
    ├── ProcessingError
    ├── FileValidationError
    ├── ConversionError
    ├── JobError
    │   ├── JobSubmissionError
    │   ├── JobTimeoutError
    │   └── JobCancelledError
    └── APIError
        └── RateLimitError
"""


class ImpulatorError(Exception):
    """Base exception for IMPULATOR.

    All custom exceptions in this application should inherit from this class.
    This allows catching all application-specific errors with a single except clause.

    Example:
        >>> try:
        ...     risky_operation()
        ... except ImpulatorError as e:
        ...     handle_error(e)
    """

    def __init__(self, message: str = "An error occurred in IMPULATOR"):
        self.message = message
        super().__init__(self.message)


# Backwards compatibility alias
MolecularCalculatorError = ImpulatorError


class InvalidSMILESError(ImpulatorError):
    """Raised when a SMILES string is invalid.

    This exception includes the invalid SMILES string for debugging purposes.

    Attributes:
        smiles: The invalid SMILES string
        message: Human-readable error message

    Example:
        >>> raise InvalidSMILESError("invalid_smiles", "Could not parse SMILES")
    """

    def __init__(self, smiles: str, message: str = None):
        self.smiles = smiles
        self.message = message or f"Invalid SMILES: {smiles}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return f"InvalidSMILESError(smiles={self.smiles!r}, message={self.message!r})"


class ProcessingError(ImpulatorError):
    """Raised when compound processing fails.

    This can occur during similarity search, efficiency metric calculation,
    or any step of the processing pipeline.

    Attributes:
        compound_name: Name of the compound that failed (optional)
        step: Processing step that failed (optional)

    Example:
        >>> raise ProcessingError(
        ...     "Failed to calculate efficiency metrics",
        ...     compound_name="Aspirin",
        ...     step="efficiency_metrics"
        ... )
    """

    def __init__(
        self,
        message: str = "Processing failed",
        compound_name: str = None,
        step: str = None
    ):
        self.compound_name = compound_name
        self.step = step
        super().__init__(message)


class FileValidationError(ImpulatorError):
    """Raised when file validation fails.

    This can occur for various reasons:
    - File is too large
    - File type is not allowed
    - File is corrupted or unreadable
    - File is empty

    Attributes:
        filename: Name of the file that failed validation (optional)

    Example:
        >>> raise FileValidationError("File exceeds maximum size of 50MB")
    """

    def __init__(self, message: str = "File validation failed", filename: str = None):
        self.filename = filename
        super().__init__(message)


class ConversionError(ImpulatorError):
    """Raised when chemical format conversion fails.

    This can occur when converting between formats like:
    - InChI to SMILES
    - InChI Key to SMILES
    - Name to SMILES

    Attributes:
        input_value: The value that failed to convert
        source_format: The source format (e.g., 'inchi_key')
        target_format: The target format (e.g., 'smiles')

    Example:
        >>> raise ConversionError(
        ...     "Could not convert InChI Key",
        ...     input_value="INVALID-KEY",
        ...     source_format="inchi_key",
        ...     target_format="smiles"
        ... )
    """

    def __init__(
        self,
        message: str = "Format conversion failed",
        input_value: str = None,
        source_format: str = None,
        target_format: str = None
    ):
        self.input_value = input_value
        self.source_format = source_format
        self.target_format = target_format
        super().__init__(message)


class APIError(ImpulatorError):
    """Raised when an external API call fails.

    This is the base class for API-related errors.

    Attributes:
        api_name: Name of the API that failed (e.g., 'ChEMBL', 'PDB')
        status_code: HTTP status code (if applicable)

    Example:
        >>> raise APIError("API request failed", api_name="ChEMBL", status_code=500)
    """

    def __init__(
        self,
        message: str = "API call failed",
        api_name: str = None,
        status_code: int = None
    ):
        self.api_name = api_name
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded.

    This exception indicates that too many requests have been made
    to an external API in a short time period.

    Attributes:
        retry_after: Seconds to wait before retrying (if known)

    Example:
        >>> raise RateLimitError(
        ...     "Rate limit exceeded for ChEMBL API",
        ...     api_name="ChEMBL",
        ...     retry_after=60
        ... )
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        api_name: str = None,
        retry_after: int = None
    ):
        self.retry_after = retry_after
        super().__init__(message, api_name=api_name)


# Job-related exceptions
class JobError(ImpulatorError):
    """Base exception for job-related errors.

    Attributes:
        job_id: ID of the job that failed (optional)

    Example:
        >>> raise JobError("Job failed", job_id="abc123")
    """

    def __init__(self, message: str = "Job error", job_id: str = None):
        self.job_id = job_id
        super().__init__(message)


class JobSubmissionError(JobError):
    """Raised when job submission fails.

    Example:
        >>> raise JobSubmissionError("Failed to submit job to backend")
    """

    def __init__(self, message: str = "Job submission failed", job_id: str = None):
        super().__init__(message, job_id=job_id)


class JobTimeoutError(JobError):
    """Raised when a job times out.

    Attributes:
        timeout_seconds: The timeout duration

    Example:
        >>> raise JobTimeoutError("Job timed out", job_id="abc123", timeout_seconds=3600)
    """

    def __init__(
        self,
        message: str = "Job timed out",
        job_id: str = None,
        timeout_seconds: int = None
    ):
        self.timeout_seconds = timeout_seconds
        super().__init__(message, job_id=job_id)


class JobCancelledError(JobError):
    """Raised when a job is cancelled.

    Example:
        >>> raise JobCancelledError("Job was cancelled by user", job_id="abc123")
    """

    def __init__(self, message: str = "Job was cancelled", job_id: str = None):
        super().__init__(message, job_id=job_id)


class BackendUnavailableError(ImpulatorError):
    """Raised when the backend API is unavailable.

    Example:
        >>> raise BackendUnavailableError("Cannot connect to backend API")
    """

    def __init__(self, message: str = "Backend API is unavailable"):
        super().__init__(message)
