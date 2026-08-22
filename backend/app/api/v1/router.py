"""
=============================================================================
ROSA Knee AI — API v1 Router
=============================================================================
Main router for API version 1. All sub-routers are registered here.

Endpoints:
  - Cases:       POST/GET/DELETE /api/v1/cases
  - Processing:  POST /api/v1/cases/{id}/preprocess, segment, reconstruct
                 GET  /api/v1/cases/{id}/results, mesh/{name}
  - Viewer:      GET  /api/v1/cases/{id}/slices/{axis}/{index}
                 GET  /api/v1/cases/{id}/volume-info
  - Model:       GET  /api/v1/model/status
=============================================================================
"""

from fastapi import APIRouter

from backend.app.schemas.responses import ModelStatusResponse
from backend.app.api.v1.cases import router as cases_router
from backend.app.api.v1.processing import router as processing_router
from backend.app.api.v1.viewer import router as viewer_router

# ---------------------------------------------------------------------------
# Create the main v1 router — all sub-routers attach here
# ---------------------------------------------------------------------------
router = APIRouter()

# Register sub-routers
router.include_router(cases_router)
router.include_router(processing_router)
router.include_router(viewer_router)


# ---------------------------------------------------------------------------
# Model status endpoint
# ---------------------------------------------------------------------------
@router.get(
    "/model/status",
    response_model=ModelStatusResponse,
    tags=["Model"],
    summary="Get segmentation model status",
)
async def get_model_status():
    """Returns the current status of the segmentation model."""
    return ModelStatusResponse(
        model_name="PlaceholderSegmentationModel",
        status="unavailable",
        message="Segmentation model has not been trained yet. "
                "This is expected during development.",
        trained=False,
    )
