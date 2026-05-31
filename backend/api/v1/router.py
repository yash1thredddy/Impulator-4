"""
API v1 router aggregator.
"""
from fastapi import APIRouter

from backend.api.v1 import health, jobs, compounds, collections

# Create main API router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(health.router)
api_router.include_router(jobs.router)
api_router.include_router(compounds.router)
api_router.include_router(collections.router)
