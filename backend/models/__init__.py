"""Models package -- ORM models and Pydantic schemas."""

from .database import Job, Compound, DeletedCompound, JobStatus, JobType
from .schemas import (
    # Enums
    DuplicateAction,
    # Job schemas
    JobCreate, BatchJobCreate, JobResponse, JobDetailResponse,
    JobProgress, ActiveJobResponse, JobListResponse,
    BatchSummary, BatchResponse,
    # Compound schemas
    CompoundBase, CompoundCreate, CompoundResponse, CompoundList,
    CompoundListItem, CompoundListResponse, CompoundDetailResponse,
    CompoundDeleteResponse, BatchDeleteResponse,
    CompoundVersionItem, CompoundVersionsResponse,
    CompoundStructure,
    # Duplicate schemas
    CheckDuplicatesRequest, CheckDuplicatesResponse,
    DuplicateMatch, InternalDuplicateMatch,
    DuplicateFoundResponse, ResolveDuplicateRequest,
    ExistingCompoundInfo,
    # Availability schemas
    CheckAvailabilityRequest, CheckAvailabilityResponse,
    CheckAvailabilityBatchRequest, CheckAvailabilityBatchResponse,
    CompoundInput, CompoundAvailability, ThresholdAvailability,
    ExistingCompoundAtThreshold,
    # Response schemas
    ErrorResponse, MessageResponse, SkipResponse,
    DeleteResponse, CancelResponse,
    FailedCompound,
    # Health schemas
    HealthResponse, ExecutorStats,
    # Typed JSON blobs
    InputParams, ResultSummary,
)

__all__ = [
    # ORM models
    "Job", "Compound", "DeletedCompound", "JobStatus", "JobType",
    # Enums
    "DuplicateAction",
    # Job schemas
    "JobCreate", "BatchJobCreate", "JobResponse", "JobDetailResponse",
    "JobProgress", "ActiveJobResponse", "JobListResponse",
    "BatchSummary", "BatchResponse",
    # Compound schemas
    "CompoundBase", "CompoundCreate", "CompoundResponse", "CompoundList",
    "CompoundListItem", "CompoundListResponse", "CompoundDetailResponse",
    "CompoundDeleteResponse", "BatchDeleteResponse",
    "CompoundVersionItem", "CompoundVersionsResponse",
    "CompoundStructure",
    # Duplicate schemas
    "CheckDuplicatesRequest", "CheckDuplicatesResponse",
    "DuplicateMatch", "InternalDuplicateMatch",
    "DuplicateFoundResponse", "ResolveDuplicateRequest",
    "ExistingCompoundInfo",
    # Availability schemas
    "CheckAvailabilityRequest", "CheckAvailabilityResponse",
    "CheckAvailabilityBatchRequest", "CheckAvailabilityBatchResponse",
    "CompoundInput", "CompoundAvailability", "ThresholdAvailability",
    "ExistingCompoundAtThreshold",
    # Response schemas
    "ErrorResponse", "MessageResponse", "SkipResponse",
    "DeleteResponse", "CancelResponse",
    "FailedCompound",
    # Health schemas
    "HealthResponse", "ExecutorStats",
    # Typed JSON blobs
    "InputParams", "ResultSummary",
]
