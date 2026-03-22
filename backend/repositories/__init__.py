"""Repository layer for centralized database access.

Exports standalone repository singletons for Jobs and Compounds.
No BaseRepository -- repositories are standalone classes using SA 2.0 select() style.
"""

from backend.repositories.compound_repository import CompoundRepository, compound_repo
from backend.repositories.job_repository import (
    DuplicateEntryError,
    JobRepository,
    ReferenceError,
    ValidationError,
    job_repo,
)

__all__ = [
    "DuplicateEntryError",
    "ReferenceError",
    "ValidationError",
    "JobRepository",
    "job_repo",
    "CompoundRepository",
    "compound_repo",
]
