"""
Shared FastAPI Annotated dependency aliases.

Centralizes dependency injection types so route handlers use concise signatures
like ``db: DbDep`` instead of ``db: Session = Depends(get_db)``.

FastAPI resolves ``Annotated`` deps to the underlying callable, so
``app.dependency_overrides[get_db]`` in tests continues to work.
"""
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.core.database import get_db
from backend.core.auth import validate_session_id
from backend.core.rate_limiter import job_rate_limit_dep, batch_rate_limit_dep

DbDep = Annotated[Session, Depends(get_db)]
SessionDep = Annotated[str, Depends(validate_session_id)]
JobRateLimit = Annotated[None, Depends(job_rate_limit_dep)]
BatchRateLimit = Annotated[None, Depends(batch_rate_limit_dep)]
