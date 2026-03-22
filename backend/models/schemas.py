"""
Pydantic schemas for API request/response validation.

Request schemas define what clients send (JobCreate, BatchJobCreate, etc.).
Response schemas define what the API returns, matching Postgres ORM models.

Enums are imported from ``backend.models.enums`` (single source of truth).
"""
import re
import uuid
from datetime import datetime
from typing import Annotated, Any
from enum import Enum

from pydantic import AfterValidator, BaseModel, Field, ConfigDict, field_validator, model_validator

from backend.models.enums import JobStatus, JobType

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


def _validate_smiles_field(v: str) -> str:
    """Shared SMILES validation logic for all schemas."""
    if not v or not v.strip():
        raise ValueError('SMILES string cannot be empty')
    v = v.strip()
    if len(v) > 5000:
        raise ValueError('SMILES too long (max 5000 characters)')
    if not SMILES_PATTERN.match(v):
        raise ValueError('SMILES contains invalid characters')
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(v)
        except Exception as e:
            raise ValueError(f'Invalid SMILES: {str(e)}')
        if mol is None:
            raise ValueError('Invalid SMILES: could not parse as a valid molecule')
    return v


def _validate_compound_name(v: str) -> str:
    """Shared compound name validation logic for all schemas."""
    if not v or not v.strip():
        raise ValueError('Compound name cannot be empty')
    v = v.strip()
    if len(v) > 255:
        raise ValueError('Compound name too long (max 255 characters)')
    if not COMPOUND_NAME_PATTERN.match(v):
        raise ValueError('Compound name contains invalid characters')
    if '..' in v or '/' in v or '\\' in v:
        raise ValueError('Compound name contains invalid path characters')
    if '\x00' in v:
        raise ValueError('Compound name contains invalid characters')
    return v


def _validate_author_name(v: str) -> str:
    """Shared author name validation logic for all schemas."""
    if not v or not v.strip():
        raise ValueError('Author name cannot be empty')
    v = v.strip()
    if len(v) > 100:
        raise ValueError('Author name too long (max 100 characters)')
    if not COMPOUND_NAME_PATTERN.match(v):
        raise ValueError('Author name contains invalid characters')
    if '\x00' in v:
        raise ValueError('Author name contains invalid characters')
    return v


# --- Reusable Field Types ---
CompoundName = Annotated[str, Field(min_length=1, max_length=255), AfterValidator(_validate_compound_name)]
SmilesString = Annotated[str, Field(min_length=1, max_length=5000), AfterValidator(_validate_smiles_field)]
AuthorName = Annotated[str, Field(min_length=1, max_length=100), AfterValidator(_validate_author_name)]
SimilarityThreshold = Annotated[int, Field(ge=40, le=100)]


# ============================================================================
# Request Schemas (client -> API)
# ============================================================================


class JobCreate(BaseModel):
    """Request schema for creating a job."""

    compound_name: CompoundName
    author_name: AuthorName
    smiles: SmilesString
    similarity_threshold: SimilarityThreshold = 90
    activity_types: list[str] | None = None
    # Session ID for user isolation (passed from frontend)
    session_id: str | None = None
    # Per-compound duplicate action (for batch processing)
    duplicate_action: str | None = Field(
        None,
        description="Action for handling duplicate: 'skip', 'replace', or 'duplicate'"
    )
    # Original compound name (for duplicates with renamed compounds)
    original_compound_name: str | None = Field(
        None,
        description="Original compound name when creating a duplicate with a new name"
    )

    @field_validator('duplicate_action')
    @classmethod
    def validate_duplicate_action(cls, v: str | None) -> str | None:
        """Validate duplicate_action is one of the allowed values."""
        if v is None:
            return v
        allowed_actions = {'skip', 'replace', 'duplicate'}
        if v not in allowed_actions:
            raise ValueError(f"duplicate_action must be one of: {', '.join(allowed_actions)}")
        return v

    @field_validator('original_compound_name')
    @classmethod
    def validate_original_compound_name(cls, v: str | None) -> str | None:
        """Validate original_compound_name for safety.

        Uses same whitelist pattern as compound_name.
        """
        if v is None:
            return v

        v = v.strip()
        if not v:
            return None

        # Length check
        if len(v) > 255:
            raise ValueError('Original compound name too long (max 255 characters)')

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

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "compound_name": "Aspirin",
                "author_name": "Dr. Smith",
                "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
                "similarity_threshold": 90,
                "activity_types": ["IC50", "Ki"],
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    )


class BatchJobCreate(BaseModel):
    """Request schema for creating a batch job."""

    compounds: list[JobCreate] = Field(..., min_length=1, max_length=1000)
    # Batch-level author_name — applied to all compounds that don't specify their own
    author_name: AuthorName | None = Field(
        None,
        description="Author name applied to all compounds in the batch. Per-compound author_name overrides this."
    )
    # Batch-level similarity_threshold — applied to compounds that don't specify their own
    similarity_threshold: SimilarityThreshold | None = Field(
        None,
        description="Default similarity threshold for all compounds. Per-compound value overrides."
    )
    # Batch-level activity_types
    activity_types: list[str] | None = Field(
        None,
        description="Default activity types for all compounds. Per-compound value overrides."
    )
    # Session ID for user isolation (applied to all jobs in batch)
    session_id: str | None = None
    # Per-compound duplicate decisions: maps compound_name -> action ('skip', 'replace', 'duplicate')
    duplicate_decisions: dict | None = Field(
        None,
        description="Dict mapping compound names to duplicate actions. Alternative to setting duplicate_action on each compound."
    )

    @field_validator('duplicate_decisions')
    @classmethod
    def validate_duplicate_decisions(cls, v: dict | None) -> dict | None:
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

            if len(compound_name) > 255:
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

    @model_validator(mode="after")
    def validate_all_smiles(self) -> "BatchJobCreate":
        """Validate all SMILES in the batch synchronously (QUAL-02).

        Returns 422 listing which entries have invalid SMILES.
        Latency of 1-5s for 1000 compounds is acceptable per CONTEXT.md.
        """
        if not RDKIT_AVAILABLE:
            return self

        invalid_entries = []
        for i, compound in enumerate(self.compounds):
            smiles = compound.smiles
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        invalid_entries.append(
                            f"Entry {i} ('{compound.compound_name}'): invalid SMILES '{smiles}'"
                        )
                except Exception:
                    invalid_entries.append(
                        f"Entry {i} ('{compound.compound_name}'): unparseable SMILES '{smiles}'"
                    )
        if invalid_entries:
            raise ValueError(
                f"Batch contains {len(invalid_entries)} invalid SMILES: " + "; ".join(invalid_entries)
            )
        return self

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Ibuprofen", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "duplicate_action": "replace"},
                ],
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "duplicate_decisions": {
                    "Quercetin": "skip",
                    "Resveratrol": "duplicate"
                },
            }
        }
    )


class CompoundStructure(BaseModel):
    """Structure data for a compound in batch duplicate check."""

    compound_name: str = Field(..., min_length=1, max_length=255)
    smiles: str | None = Field(None, description="SMILES string for the compound")
    inchi: str | None = Field(None, description="InChI string for the compound (converted to SMILES if smiles not provided)")
    inchikey: str | None = Field(
        None,
        description="InChIKey for the compound (used directly when SMILES/InChI are unavailable)"
    )


class CheckDuplicatesRequest(BaseModel):
    """Request schema for checking duplicate compounds.

    Supports two modes:
    1. Name-only check (legacy): Just provide compound_names
    2. Structure-based check (recommended): Provide compounds with SMILES/InChI/InChIKey for InChIKey-based detection
    """

    # Legacy: name-only list (for backward compatibility)
    compound_names: list[str] | None = Field(None, max_length=1000)
    # New: compounds with structure data for InChIKey-based duplicate detection
    compounds: list[CompoundStructure] | None = Field(None, max_length=1000)
    # Optional config context (enables config-aware duplicate messaging in batch mode)
    similarity_threshold: SimilarityThreshold | None = 90
    activity_types: list[str] | None = None

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "compounds": [
                    {"compound_name": "Aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
                    {"compound_name": "Quercetin", "smiles": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C(O)=C2O"},
                    {"compound_name": "Unknown", "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"},
                    {"compound_name": "KnownByKey", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"}
                ]
            }
        }
    )


class CheckAvailabilityRequest(BaseModel):
    """Pre-submission check for ChEMBL data availability."""
    smiles: SmilesString
    similarity_threshold: SimilarityThreshold = 90
    activity_types: list[str] | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class CompoundInput(BaseModel):
    """Compound identifier for availability checks."""
    compound_name: CompoundName
    smiles: SmilesString
    threshold: SimilarityThreshold | None = Field(None, description="Per-compound threshold override")


class CheckAvailabilityBatchRequest(BaseModel):
    """Batch availability check -- one probe per compound."""
    compounds: list[CompoundInput] = Field(..., min_length=1, max_length=1000)
    similarity_threshold: SimilarityThreshold = 90
    activity_types: list[str] | None = None


class ResolveDuplicateRequest(BaseModel):
    """Request to resolve a duplicate compound situation."""

    action: "DuplicateAction"
    smiles: SmilesString
    compound_name: CompoundName
    author_name: AuthorName
    existing_entry_id: uuid.UUID | None = None
    new_compound_name: str | None = Field(None, description="New name if user wants to change it (for exact duplicates)")
    # Keep threshold range aligned with JobCreate and frontend slider (40-100).
    similarity_threshold: SimilarityThreshold = 90
    activity_types: list[str] | None = None
    session_id: str | None = None

    @field_validator('new_compound_name')
    @classmethod
    def validate_new_compound_name(cls, v: str | None) -> str | None:
        """Validate new compound name for safety."""
        if v is None:
            return v
        v = v.strip()
        if not v:
            return None
        if len(v) > 255:
            raise ValueError('New compound name too long (max 255 characters)')
        if not COMPOUND_NAME_PATTERN.match(v):
            raise ValueError('New compound name contains invalid characters')
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('New compound name contains invalid path characters')
        if '\x00' in v:
            raise ValueError('New compound name contains invalid characters')
        return v

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "action": "duplicate",
                "smiles": "O=C1C(O)=C(O)C(=O)C2=C1C=C(O)C=C2O",
                "compound_name": "Quercetin_v2",
                "existing_entry_id": "3a4f8c9e-1b2d-4e5f-9a1c-2d3e4f5a6b7c",
            }
        }
    )


# ============================================================================
# DuplicateAction Enum (request-side, not in Postgres)
# ============================================================================


class DuplicateAction(str, Enum):
    """User action for handling duplicate compounds.

    This is a request-side enum, not stored in Postgres as a DB enum type.
    Kept in schemas.py rather than enums.py.
    """

    REPLACE = "replace"       # Overwrite existing compound
    DUPLICATE = "duplicate"   # Save as new with duplicate tag
    SKIP = "skip"             # Don't process


# ============================================================================
# Response Schemas (API -> client)
# ============================================================================


class JobResponse(BaseModel):
    """Response schema for job status."""

    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    compound_name: str
    smiles: str | None = None
    similarity_threshold: int = 90
    activity_types: list[str] | None = None
    progress: float = Field(ge=0, le=100)
    current_step: str | None = None
    error_message: str | None = None
    session_id: uuid.UUID | None = None
    batch_id: uuid.UUID | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class JobDetailResponse(BaseModel):
    """Detailed job response with direct column access."""

    id: uuid.UUID
    job_type: JobType | None = None
    status: JobStatus | None = None
    session_id: uuid.UUID | None = None
    batch_id: uuid.UUID | None = None
    compound_name: str | None = None
    smiles: str | None = None
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    progress: float | None = None
    current_step: str | None = None
    error_message: str | None = None
    result_summary: dict[str, Any] | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    cancelled_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ActiveJobResponse(BaseModel):
    """Response schema for active jobs in sidebar."""

    id: uuid.UUID
    status: JobStatus
    progress: float
    current_step: str | None = None
    compound_name: str | None = None
    batch_id: uuid.UUID | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None
    entry_id: uuid.UUID | None = None
    storage_path: str | None = None
    error_message: str | None = None
    cascade_results: list[dict] | None = None  # [{threshold: 80, count: 5}, ...]
    system_status: str | None = None  # "healthy" | "degraded" | None (D-12)
    input_params: dict | None = None  # Compound params for resubmission (D-70)

    model_config = ConfigDict(from_attributes=True)


class JobProgress(BaseModel):
    """Schema for job progress updates (polling)."""

    job_id: uuid.UUID
    status: JobStatus
    progress: float
    current_step: str | None = None
    message: str | None = None


class CompoundListItem(BaseModel):
    """Single compound in paginated list."""
    model_config = ConfigDict(from_attributes=True)

    entry_id: uuid.UUID
    compound_name: str
    smiles: str | None = None
    inchikey: str | None = None
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    similar_compounds: int | None = None
    total_activities: int | None = None
    parent_id: uuid.UUID | None = None
    version: int = 1
    config_diff: dict | None = None
    processed_at: datetime | None = None
    storage_path: str | None = None
    chembl_id: str | None = None
    imp_candidates: int | None = None
    imp_score: float | None = None
    num_outliers: int | None = None
    qed: float | None = None
    is_duplicate: bool = False       # Computed: parent_id is not None
    parent_name: str | None = None   # Resolved via JOIN


class CompoundListResponse(BaseModel):
    """Paginated compound list -- consistent envelope per ARCH-15."""
    items: list[CompoundListItem]
    total: int
    page: int
    page_size: int
    pages: int


class CompoundDetailResponse(BaseModel):
    """Single compound metadata response."""
    model_config = ConfigDict(from_attributes=True)

    entry_id: uuid.UUID
    compound_name: str
    smiles: str | None = None
    inchikey: str | None = None
    inchikey_structure_key: str | None = None
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    similar_compounds: int | None = None
    total_activities: int | None = None
    parent_id: uuid.UUID | None = None
    version: int = 1
    config_diff: dict | None = None
    processed_at: datetime | None = None
    storage_path: str | None = None
    chembl_id: str | None = None
    imp_candidates: int | None = None
    imp_score: float | None = None
    num_outliers: int | None = None
    qed: float | None = None
    author_name: str | None = None


class CompoundVersionItem(BaseModel):
    """A single version (structural sibling) of a compound."""

    entry_id: uuid.UUID
    compound_name: str
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    imp_score: float | None = None
    qed: float | None = None
    similar_compounds: int | None = None
    total_activities: int | None = None
    parent_id: uuid.UUID | None = None
    version: int = 1
    config_diff: dict | None = None
    parent_name: str | None = None   # Resolved via JOIN or lookup
    author_name: str | None = None
    processed_at: datetime | None = None
    storage_path: str | None = None
    is_original: bool = False  # Computed, not stored
    is_current: bool = False   # Computed, not stored

    model_config = ConfigDict(from_attributes=True)


class CompoundVersionsResponse(BaseModel):
    """Response for compound versions endpoint."""

    versions: list[CompoundVersionItem] = []
    current_entry_id: uuid.UUID


class DuplicateMatch(BaseModel):
    """Information about a duplicate match found by InChIKey."""

    compound_name: str = Field(..., description="Name of compound in the request")
    inchikey: str | None = Field(None, description="Generated InChIKey")
    existing_compound_name: str = Field(..., description="Name of existing compound with same structure")
    existing_entry_id: uuid.UUID | None = Field(None, description="Entry ID of existing compound")
    match_type: str = Field(..., description="'exact' (same name+structure), 'structure_only' (different name, same structure)")
    config_match: str | None = Field(
        None,
        description="'identical', 'different_threshold', 'different_activities', or 'different_both'"
    )
    config_diff: dict | None = Field(
        None,
        description="Config comparison details when config_match != 'identical'"
    )
    existing_similarity_threshold: int | None = Field(
        None,
        description="Similarity threshold used by existing compound"
    )
    existing_activity_types: list[str] | None = Field(
        None,
        description="Normalized activity types used by existing compound"
    )
    existing_author_name: str | None = Field(
        None,
        description="Author name of the existing compound record"
    )
    existing_processed_at: datetime | None = Field(
        None,
        description="Timestamp when existing compound was processed"
    )


class InternalDuplicateMatch(BaseModel):
    """Information about a duplicate found within the submitted payload itself."""

    compound_name: str = Field(..., description="Duplicate compound name in the submitted payload")
    duplicate_of: str = Field(..., description="First-seen compound name this duplicates in the same payload")
    match_type: str = Field(..., description="'exact' (same name+structure) or 'structure_only' (same structure)")
    inchikey: str | None = Field(None, description="Generated InChIKey when available")


class CheckDuplicatesResponse(BaseModel):
    """Response schema for duplicate check."""

    existing: list[str] = Field(default_factory=list, description="Compounds that already have results (by name)")
    processing: list[str] = Field(default_factory=list, description="Compounds currently being processed")
    new: list[str] = Field(default_factory=list, description="Compounds that are new")
    # InChIKey-based duplicate matches (more accurate than name-based)
    structure_matches: list[DuplicateMatch] = Field(
        default_factory=list,
        description="Compounds that match existing compounds by InChIKey (structure)"
    )
    internal_duplicates: list[InternalDuplicateMatch] = Field(
        default_factory=list,
        description="Compounds duplicated within the submitted payload itself (same name or structure)"
    )
    # Suggested version names for existing compounds (computed from full database state)
    suggested_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of existing compound name -> suggested next version name (e.g., 'Aspirin' -> 'Aspirin_v3')"
    )


class ExistingCompoundAtThreshold(BaseModel):
    """An existing compound that matches by InChIKey, with config comparison."""
    entry_id: uuid.UUID
    compound_name: str
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    config_match: str  # 'identical', 'different_threshold', 'different_activities', 'different_both'
    config_diff: dict | None = None
    imp_score: float | None = None
    processed_at: datetime | None = None
    author_name: str | None = None


class ThresholdAvailability(BaseModel):
    """Data availability at a single threshold."""
    threshold: int
    count: int  # 0 = no data


class CompoundAvailability(BaseModel):
    """Availability result for a single compound."""
    compound_name: str
    smiles: str
    available: bool  # True = data at requested threshold
    count_at_threshold: int
    thresholds: list[ThresholdAvailability] = []
    existing_compounds: list[ExistingCompoundAtThreshold] = []
    has_any_data: bool = True  # False = no data at ANY threshold


class CheckAvailabilityResponse(BaseModel):
    """Single compound availability response."""
    result: CompoundAvailability


class CheckAvailabilityBatchResponse(BaseModel):
    """Batch availability response."""
    results: list[CompoundAvailability] = []
    available_count: int = 0
    unavailable_count: int = 0
    no_data_count: int = 0


class ExistingCompoundInfo(BaseModel):
    """Information about an existing compound (for duplicate detection)."""

    entry_id: uuid.UUID | None = None
    compound_name: str
    inchikey: str | None = None
    processed_at: datetime | None = None
    similarity_threshold: int | None = None
    activity_types: list[str] | None = None
    author_name: str | None = None


class DuplicateFoundResponse(BaseModel):
    """Response when a duplicate compound is detected during job submission."""

    status: str = "duplicate_found"
    duplicate_type: str = Field(..., description="'exact' if both structure and name match, 'structure_only' if only structure matches")
    config_match: str = Field(default="identical", description="'identical', 'different_threshold', 'different_activities', 'different_both'")
    existing_compound: ExistingCompoundInfo
    submitted: dict = Field(..., description="Info about the submitted compound")
    suggested_name: str | None = Field(None, description="Suggested unique name for duplicate (e.g., 'Quercetin_v3')")
    config_diff: dict | None = Field(None, description="Config comparison details when config_match != 'identical'")


class BatchSummary(BaseModel):
    """Summary of a batch of jobs for sidebar display."""

    batch_id: uuid.UUID
    total_jobs: int
    completed: int
    processing: int
    pending: int
    failed: int
    cancelled: int = 0
    overall_progress: float = Field(ge=0, le=100)
    created_at: datetime | None = None
    # Sample of compound names in this batch
    compound_names: list[str] = []


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    items: list[JobResponse]
    total: int
    page: int
    page_size: int
    pages: int


class BatchResponse(BaseModel):
    """Response for batch job creation."""

    batch_id: uuid.UUID
    jobs: list[JobResponse]
    skipped_existing: list[str] = []
    skipped_processing: list[str] = []
    skipped_internal_duplicates: list[str] = []
    replaced: list[str] = []  # Compounds that were replaced (existing deleted)
    failed_compounds: list["FailedCompound"] = []  # Compounds that failed during job creation
    total_submitted: int
    total_skipped: int
    message: str | None = None


class FailedCompound(BaseModel):
    """Information about a compound that failed during batch job creation."""

    compound_name: str
    error: str


class DeleteResponse(BaseModel):
    """Response for job deletion."""

    message: str
    job_id: uuid.UUID
    compound_name: str | None = None


class CancelResponse(BaseModel):
    """Response for batch cancellation."""

    batch_id: uuid.UUID
    cancelled_count: int
    message: str


class CompoundDeleteResponse(BaseModel):
    """Response for single compound deletion."""
    status: str = "deleted"
    entry_id: uuid.UUID
    message: str


class BatchDeleteResponse(BaseModel):
    """Response for batch compound deletion."""
    status: str = "completed"
    deleted: list[str]
    failed: list[dict[str, str]] = Field(default_factory=list)
    total_deleted: int
    total_failed: int


# ============================================================================
# Health Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    database: bool
    db_latency_ms: float | None = None  # Database round-trip time in ms
    azure_configured: bool
    active_jobs: int
    max_concurrent_jobs: int = 10
    timestamp: datetime


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_code: str | None = None
    request_id: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Compound not found",
                "error_code": "NOT_FOUND",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
    )


# ============================================================================
# Generic Response Schemas
# ============================================================================


class MessageResponse(BaseModel):
    """Generic message response."""

    status: str
    message: str


class SkipResponse(MessageResponse):
    """Response when a compound is skipped."""

    compound_name: str | None = None
