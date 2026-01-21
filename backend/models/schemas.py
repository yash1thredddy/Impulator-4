"""
Pydantic schemas for API request/response validation.
"""
import re
from datetime import datetime
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator

# Try to import RDKit for SMILES validation
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Whitelist pattern for SMILES - only valid SMILES characters
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9@+\-\[\]\(\)\\/#=%\.\*\:]+$')

# Whitelist pattern for compound names - safe characters only
COMPOUND_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9\-_\s\(\)\[\]',\.]+$")


# Enums
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    SINGLE = "single"
    BATCH = "batch"


# Job Schemas
class JobCreate(BaseModel):
    """Request schema for creating a job."""

    compound_name: str = Field(..., min_length=1, max_length=100)
    smiles: str = Field(..., min_length=1, max_length=5000)
    similarity_threshold: int = Field(default=90, ge=50, le=100)
    activity_types: Optional[List[str]] = None
    # Session ID for user isolation (passed from frontend)
    session_id: Optional[str] = None
    # Per-compound duplicate action (for batch processing)
    duplicate_action: Optional[str] = Field(
        None,
        description="Action for handling duplicate: 'skip', 'replace', or 'duplicate'"
    )
    # Original compound name (for duplicates with renamed compounds)
    original_compound_name: Optional[str] = Field(
        None,
        description="Original compound name when creating a duplicate with a new name"
    )

    @field_validator('duplicate_action')
    @classmethod
    def validate_duplicate_action(cls, v: Optional[str]) -> Optional[str]:
        """Validate duplicate_action is one of the allowed values."""
        if v is None:
            return v
        allowed_actions = {'skip', 'replace', 'duplicate'}
        if v not in allowed_actions:
            raise ValueError(f"duplicate_action must be one of: {', '.join(allowed_actions)}")
        return v

    @field_validator('original_compound_name')
    @classmethod
    def validate_original_compound_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate original_compound_name for safety.

        Uses same whitelist pattern as compound_name.
        """
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        # Length check
        if len(v) > 100:
            raise ValueError('Original compound name too long (max 100 characters)')

        # Whitelist pattern - only safe characters
        if not COMPOUND_NAME_PATTERN.match(v):
            raise ValueError('Original compound name contains invalid characters')

        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Original compound name contains invalid path characters')

        # Check for null bytes
        if '\x00' in v:
            raise ValueError('Original compound name contains invalid characters')

        return v

    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Validate SMILES string format and chemical validity.

        Uses whitelist pattern to only allow valid SMILES characters,
        preventing injection attacks. RDKit validation is required if available.
        """
        if not v or not v.strip():
            raise ValueError('SMILES string cannot be empty')

        v = v.strip()

        # Length check
        if len(v) > 2000:
            raise ValueError('SMILES too long (max 2000 characters)')

        # Whitelist pattern check - only valid SMILES characters
        if not SMILES_PATTERN.match(v):
            raise ValueError('SMILES contains invalid characters')

        # RDKit validation (required, not optional)
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(v)
                if mol is None:
                    raise ValueError('Invalid SMILES: could not parse as a valid molecule')
            except Exception as e:
                raise ValueError(f'Invalid SMILES: {str(e)}')

        return v

    @field_validator('compound_name')
    @classmethod
    def validate_compound_name(cls, v: str) -> str:
        """Validate compound name for safety.

        Uses whitelist pattern and checks for path traversal attempts.
        """
        if not v or not v.strip():
            raise ValueError('Compound name cannot be empty')

        v = v.strip()

        # Length check
        if len(v) > 100:
            raise ValueError('Compound name too long (max 100 characters)')

        # Whitelist pattern - only safe characters
        if not COMPOUND_NAME_PATTERN.match(v):
            raise ValueError('Compound name contains invalid characters')

        # Check for path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Compound name contains invalid path characters')

        # Check for null bytes
        if '\x00' in v:
            raise ValueError('Compound name contains invalid characters')

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compound_name": "Aspirin",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "similarity_threshold": 90,
                "activity_types": ["IC50", "Ki"],
                "session_id": "abc123-session-uuid",
            }
        }
    )


class BatchJobCreate(BaseModel):
    """Request schema for creating a batch job."""

    compounds: List[JobCreate] = Field(..., min_length=1, max_length=100)
    # Session ID for user isolation (applied to all jobs in batch)
    session_id: Optional[str] = None
    # Per-compound duplicate decisions: maps compound_name -> action ('skip', 'replace', 'duplicate')
    duplicate_decisions: Optional[dict] = Field(
        None,
        description="Dict mapping compound names to duplicate actions. Alternative to setting duplicate_action on each compound."
    )

    @field_validator('duplicate_decisions')
    @classmethod
    def validate_duplicate_decisions(cls, v: Optional[dict]) -> Optional[dict]:
        """Validate duplicate_decisions dict structure and values.

        Ensures all keys are valid compound names and all values are valid actions.
        """
        if v is None:
            return v

        if not isinstance(v, dict):
            raise ValueError('duplicate_decisions must be a dictionary')

        allowed_actions = {'skip', 'replace', 'duplicate'}
        validated = {}

        for compound_name, action in v.items():
            # Validate key (compound name)
            if not isinstance(compound_name, str):
                raise ValueError('duplicate_decisions keys must be strings')

            compound_name = compound_name.strip()
            if not compound_name:
                raise ValueError('duplicate_decisions keys cannot be empty')

            if len(compound_name) > 100:
                raise ValueError(f'Compound name too long: {compound_name[:20]}...')

            if not COMPOUND_NAME_PATTERN.match(compound_name):
                raise ValueError(f'Invalid compound name in duplicate_decisions: {compound_name}')

            # Validate value (action)
            if not isinstance(action, str):
                raise ValueError(f'Action for {compound_name} must be a string')

            if action not in allowed_actions:
                raise ValueError(f"Action for {compound_name} must be one of: {', '.join(allowed_actions)}")

            validated[compound_name] = action

        return validated

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "duplicate_action": "replace"},
                ],
                "session_id": "abc123-session-uuid",
                "duplicate_decisions": {
                    "Quercetin": "skip",
                    "Resveratrol": "duplicate"
                },
            }
        }
    )


class JobResponse(BaseModel):
    """Response schema for job status."""

    id: str
    job_type: JobType
    status: JobStatus
    progress: float = Field(ge=0, le=100)
    current_step: Optional[str] = None
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Session and batch tracking
    session_id: Optional[str] = None
    batch_id: Optional[str] = None
    # Extracted from input_params for convenience
    compound_name: Optional[str] = None
    smiles: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class JobProgress(BaseModel):
    """Schema for job progress updates (polling)."""

    job_id: str
    status: JobStatus
    progress: float
    current_step: Optional[str] = None
    message: Optional[str] = None


class ActiveJobResponse(BaseModel):
    """Response schema for active jobs in sidebar."""

    id: str
    status: JobStatus
    progress: float
    current_step: Optional[str] = None
    compound_name: Optional[str] = None
    batch_id: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CompoundStructure(BaseModel):
    """Structure data for a compound in batch duplicate check."""

    compound_name: str = Field(..., min_length=1, max_length=100)
    smiles: Optional[str] = Field(None, description="SMILES string for the compound")
    inchi: Optional[str] = Field(None, description="InChI string for the compound (converted to SMILES if smiles not provided)")


class CheckDuplicatesRequest(BaseModel):
    """Request schema for checking duplicate compounds.

    Supports two modes:
    1. Name-only check (legacy): Just provide compound_names
    2. Structure-based check (recommended): Provide compounds with SMILES/InChI for InChIKey-based detection
    """

    # Legacy: name-only list (for backward compatibility)
    compound_names: Optional[List[str]] = Field(None, max_length=1000)
    # New: compounds with structure data for InChIKey-based duplicate detection
    compounds: Optional[List[CompoundStructure]] = Field(None, max_length=1000)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Quercetin", "smiles": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C(O)=C2O"},
                    {"compound_name": "Unknown", "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"}
                ]
            }
        }
    )


class DuplicateMatch(BaseModel):
    """Information about a duplicate match found by InChIKey."""

    compound_name: str = Field(..., description="Name of compound in the request")
    inchikey: Optional[str] = Field(None, description="Generated InChIKey")
    existing_compound_name: str = Field(..., description="Name of existing compound with same structure")
    existing_entry_id: Optional[str] = Field(None, description="Entry ID of existing compound")
    match_type: str = Field(..., description="'exact' (same name+structure), 'structure_only' (different name, same structure)")


class CheckDuplicatesResponse(BaseModel):
    """Response schema for duplicate check."""

    existing: List[str] = Field(default_factory=list, description="Compounds that already have results (by name)")
    processing: List[str] = Field(default_factory=list, description="Compounds currently being processed")
    new: List[str] = Field(default_factory=list, description="Compounds that are new")
    # InChIKey-based duplicate matches (more accurate than name-based)
    structure_matches: List[DuplicateMatch] = Field(
        default_factory=list,
        description="Compounds that match existing compounds by InChIKey (structure)"
    )


class DuplicateAction(str, Enum):
    """User action for handling duplicate compounds."""

    REPLACE = "replace"       # Overwrite existing compound
    DUPLICATE = "duplicate"   # Save as new with duplicate tag
    SKIP = "skip"             # Don't process


class ExistingCompoundInfo(BaseModel):
    """Information about an existing compound (for duplicate detection)."""

    entry_id: Optional[str] = None
    compound_name: str
    inchikey: Optional[str] = None
    processed_at: Optional[str] = None


class DuplicateFoundResponse(BaseModel):
    """Response when a duplicate compound is detected during job submission."""

    status: str = "duplicate_found"
    duplicate_type: str = Field(..., description="'exact' if both structure and name match, 'structure_only' if only structure matches")
    existing_compound: ExistingCompoundInfo
    submitted: dict = Field(..., description="Info about the submitted compound")
    suggested_name: Optional[str] = Field(None, description="Suggested unique name for duplicate (e.g., 'Quercetin_v3')")


class ResolveDuplicateRequest(BaseModel):
    """Request to resolve a duplicate compound situation."""

    action: DuplicateAction
    smiles: str
    compound_name: str
    existing_entry_id: Optional[str] = None
    new_compound_name: Optional[str] = Field(None, description="New name if user wants to change it (for exact duplicates)")
    similarity_threshold: int = Field(default=90, ge=70, le=100)
    activity_types: Optional[List[str]] = None
    session_id: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "action": "duplicate",
                "smiles": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C=C2O",
                "compound_name": "Quercetin_v2",
                "existing_entry_id": "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c",
            }
        }
    )


class BatchSummary(BaseModel):
    """Summary of a batch of jobs for sidebar display."""

    batch_id: str
    total_jobs: int
    completed: int
    processing: int
    pending: int
    failed: int
    cancelled: int = 0
    overall_progress: float = Field(ge=0, le=100)
    created_at: Optional[datetime] = None
    # Sample of compound names in this batch
    compound_names: List[str] = []


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    items: List[JobResponse]
    total: int
    page: int
    page_size: int
    pages: int


# Compound Schemas
class CompoundBase(BaseModel):
    """Base compound schema."""

    compound_name: str
    chembl_id: Optional[str] = None
    smiles: Optional[str] = None


class CompoundCreate(CompoundBase):
    """Schema for creating a compound entry."""

    total_activities: int = 0
    imp_candidates: int = 0
    avg_oqpla_score: Optional[float] = None
    storage_path: Optional[str] = None


class CompoundResponse(CompoundBase):
    """Response schema for compound listing."""

    id: int
    total_activities: int
    imp_candidates: int
    avg_oqpla_score: Optional[float] = None
    storage_path: Optional[str] = None
    processed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CompoundList(BaseModel):
    """Paginated list of compounds."""

    items: List[CompoundResponse]
    total: int
    page: int
    page_size: int
    pages: int


# Health Schemas
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    database: bool
    azure_configured: bool
    executor_active_jobs: int
    timestamp: datetime


class ExecutorStats(BaseModel):
    """Executor statistics response."""

    max_workers: int
    active_jobs: int
    has_capacity: bool
    job_ids: List[str]


# Error Schemas
class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Compound not found",
                "error_code": "NOT_FOUND",
            }
        }
    )


# Consistent Response Schemas
class MessageResponse(BaseModel):
    """Generic message response."""

    status: str
    message: str


class SkipResponse(MessageResponse):
    """Response when a compound is skipped."""

    compound_name: Optional[str] = None


class BatchResponse(BaseModel):
    """Response for batch job creation."""

    batch_id: str
    jobs: List[JobResponse]
    skipped_existing: List[str] = []
    skipped_processing: List[str] = []
    replaced: List[str] = []  # Compounds that were replaced (existing deleted)
    total_submitted: int
    total_skipped: int
    message: Optional[str] = None


class DeleteResponse(BaseModel):
    """Response for job deletion."""

    message: str
    job_id: str
    compound_name: Optional[str] = None


class CancelResponse(BaseModel):
    """Response for batch cancellation."""

    batch_id: str
    cancelled_count: int
    message: str
