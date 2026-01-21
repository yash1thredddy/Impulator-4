"""
API v1 router aggregator.
"""
from fastapi import APIRouter

from backend.api.v1 import health, jobs, compounds

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health.router)
api_router.include_router(jobs.router)
api_router.include_router(compounds.router)
