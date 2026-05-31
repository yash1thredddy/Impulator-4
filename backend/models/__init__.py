"""Models package -- Postgres ORM models (v2.2+) and Pydantic schemas."""

from backend.models._pg_base import PGBase

# ORM models
from backend.models.enums import JobStatus, JobType, AuditEventType
from backend.models.job import Job
from backend.models.compound import Compound
from backend.models.deleted_compound import DeletedCompound
from backend.models.audit import AuditEvent
from backend.models.collection import Collection

# Pydantic schemas
from backend.models.schemas import (
    # Enums
    DuplicateAction,
    # Job schemas
    JobCreate, BatchJobCreate, JobResponse, JobDetailResponse,
    JobProgress, ActiveJobResponse, JobListResponse,
    BatchSummary, BatchResponse,
    # Compound schemas
    CompoundListItem, CompoundListResponse, CompoundDetailResponse,
    CompoundDeleteResponse, BatchDeleteResponse,
    CompoundVersionItem, CompoundVersionsResponse,
    CompoundStructure,
    # Collection schemas
    CollectionMember, CollectionJobCreate, CollectionSummary,
    CollectionResponse, CollectionDetailResponse, CollectionListResponse,
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
    HealthResponse,
)

__all__ = [
    # Base
    "PGBase",
    # ORM models
    "Job", "Compound", "DeletedCompound", "AuditEvent", "Collection",
    # Enums
    "JobStatus", "JobType", "AuditEventType", "DuplicateAction",
    # Job schemas
    "JobCreate", "BatchJobCreate", "JobResponse", "JobDetailResponse",
    "JobProgress", "ActiveJobResponse", "JobListResponse",
    "BatchSummary", "BatchResponse",
    # Compound schemas
    "CompoundListItem", "CompoundListResponse", "CompoundDetailResponse",
    "CompoundDeleteResponse", "BatchDeleteResponse",
    "CompoundVersionItem", "CompoundVersionsResponse",
    "CompoundStructure",
    # Collection schemas
    "CollectionMember", "CollectionJobCreate", "CollectionSummary",
    "CollectionResponse", "CollectionDetailResponse", "CollectionListResponse",
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
    "HealthResponse",
]
