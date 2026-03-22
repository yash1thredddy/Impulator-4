"""Services package exports.

Avoid exporting singleton instances under names that collide with submodules
(`job_service`, `compound_service`). Otherwise, dotted imports like
`import backend.services.job_service as js_mod` can resolve to the singleton
object instead of the module, depending on import order.
"""

from backend.services.job_service import JobService
from backend.services.compound_service import (
    process_compound_job,
    cleanup_stale_folders,
    scan_recovery_markers,
)

# Singleton instances are still exported, but under non-shadowing names.
from backend.services.job_service import job_service as job_service_instance

__all__ = [
    "JobService",
    "process_compound_job",
    "cleanup_stale_folders",
    "scan_recovery_markers",
    "job_service_instance",
]
