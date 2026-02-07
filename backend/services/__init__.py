"""Services package exports.

Avoid exporting singleton instances under names that collide with submodules
(`job_service`, `compound_service`). Otherwise, dotted imports like
`import backend.services.job_service as js_mod` can resolve to the singleton
object instead of the module, depending on import order.
"""

from backend.services.job_service import JobService
from backend.services.compound_service import CompoundService, process_compound_job

# Singleton instances are still exported, but under non-shadowing names.
from backend.services.job_service import job_service as job_service_instance
from backend.services.compound_service import compound_service as compound_service_instance

__all__ = [
    "JobService",
    "CompoundService",
    "process_compound_job",
    "job_service_instance",
    "compound_service_instance",
]
