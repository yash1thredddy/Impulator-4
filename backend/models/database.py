"""
SQLAlchemy ORM models for Jobs and Compounds.
"""

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, Enum, Boolean, UniqueConstraint
from sqlalchemy.sql import func
import enum

from backend.core.database import Base


class JobStatus(str, enum.Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    """Job type enumeration."""

    SINGLE = "single"
    BATCH = "batch"


class Job(Base):
    """Job tracking table."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)  # UUID
    job_type = Column(Enum(JobType), nullable=False, default=JobType.SINGLE)
    status = Column(Enum(JobStatus), nullable=False, default=JobStatus.PENDING)

    # Session isolation - each browser session has a unique ID
    # This ensures users only see their own jobs
    session_id = Column(String(36), nullable=True, index=True)

    # Batch grouping - links jobs submitted together
    # Allows batch cancellation and grouped display
    batch_id = Column(String(36), nullable=True, index=True)

    # Idempotency key for safe retries - prevents duplicate job creation
    # Unique per session to allow same key across different sessions
    idempotency_key = Column(String(64), nullable=True, index=True)

    # Input parameters (JSON)
    input_params = Column(Text, nullable=True)

    # Progress tracking
    progress = Column(Float, default=0.0)
    current_step = Column(String(255), nullable=True)

    # Results
    result_path = Column(String(500), nullable=True)
    result_summary = Column(Text, nullable=True)  # JSON

    # Error handling
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Unique constraint for idempotency: same key per session = same job
    __table_args__ = (
        UniqueConstraint('session_id', 'idempotency_key', name='uix_job_session_idempotency'),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type.value if self.job_type else None,
            "status": self.status.value if self.status else None,
            "session_id": self.session_id,
            "batch_id": self.batch_id,
            "progress": self.progress,
            "current_step": self.current_step,
            "result_path": self.result_path,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Compound(Base):
    """Compound metadata table for fast listing."""

    __tablename__ = "compounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(36), unique=True, nullable=True, index=True)  # UUID for unique identification
    compound_name = Column(String(255), unique=False, nullable=False, index=True)  # NOT unique - allows duplicates
    chembl_id = Column(String(50), nullable=True, index=True)
    smiles = Column(Text, nullable=True)

    # InChIKey for duplicate detection (NOT unique - allows tagged duplicates)
    inchikey = Column(String(27), nullable=True, index=True)
    canonical_smiles = Column(Text, nullable=True)

    # Duplicate tracking
    is_duplicate = Column(Boolean, default=False, index=True)
    duplicate_of = Column(String(36), nullable=True)  # Reference to original entry_id

    # Summary statistics
    total_activities = Column(Integer, default=0)
    imp_candidates = Column(Integer, default=0)
    imp_score = Column(Float, nullable=True)

    # Additional summary fields (for home page display without ZIP download)
    similarity_threshold = Column(Integer, default=90)  # Similarity threshold % used for search
    activity_types = Column(Text, nullable=True)  # Comma-separated activity types: "IC50,Ki,Kd,EC50"
    qed = Column(Float, nullable=True)  # Average QED score
    num_outliers = Column(Integer, default=0)  # Number of outliers detected

    # Author attribution
    author_name = Column(String(100), nullable=True)

    # Storage location
    storage_path = Column(String(500), nullable=True)

    # Timestamps
    processed_at = Column(DateTime, default=func.now())

    # Note: InChIKey is NOT unique - duplicates are allowed and tagged via is_duplicate flag
    # Uniqueness is enforced via entry_id (UUID) instead
    # Duplicate detection is done at application level before job creation

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "entry_id": self.entry_id,
            "compound_name": self.compound_name,
            "chembl_id": self.chembl_id,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "canonical_smiles": self.canonical_smiles,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "total_activities": self.total_activities,
            "imp_candidates": self.imp_candidates,
            "imp_score": self.imp_score,
            "similarity_threshold": self.similarity_threshold,
            "activity_types": self.activity_types,
            "qed": self.qed,
            "num_outliers": self.num_outliers,
            "author_name": self.author_name,
            "storage_path": self.storage_path,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }


class DeletedCompound(Base):
    """Audit table for deleted compounds.

    Maintains a log of all deleted compounds for:
    - Audit trail
    - Recovery if needed
    - Preventing orphaned storage files
    """

    __tablename__ = "deleted_compounds"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Original compound data (copied from Compound table)
    original_id = Column(Integer, nullable=False)
    entry_id = Column(String(36), nullable=True, index=True)
    compound_name = Column(String(255), nullable=False, index=True)
    chembl_id = Column(String(50), nullable=True)
    smiles = Column(Text, nullable=True)
    inchikey = Column(String(27), nullable=True)
    author_name = Column(String(100), nullable=True)

    # Duplicate info
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(String(36), nullable=True)

    # Analysis config (for audit trail)
    activity_types = Column(Text, nullable=True)

    # Storage info (for cleanup verification)
    storage_path = Column(String(500), nullable=True)

    # Deletion metadata
    deleted_at = Column(DateTime, default=func.now(), nullable=False)
    deleted_by_session = Column(String(36), nullable=True)  # Session ID that deleted
    deleted_by_job_id = Column(String(36), nullable=True)  # Job ID if delete was via job
    deletion_reason = Column(String(255), nullable=True)  # e.g., "user_request", "replaced"

    # Original timestamps
    original_processed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "original_id": self.original_id,
            "entry_id": self.entry_id,
            "compound_name": self.compound_name,
            "chembl_id": self.chembl_id,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "author_name": self.author_name,
            "is_duplicate": self.is_duplicate,
            "duplicate_of": self.duplicate_of,
            "activity_types": self.activity_types,
            "storage_path": self.storage_path,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "deleted_by_session": self.deleted_by_session,
            "deletion_reason": self.deletion_reason,
            "original_processed_at": self.original_processed_at.isoformat() if self.original_processed_at else None,
        }
