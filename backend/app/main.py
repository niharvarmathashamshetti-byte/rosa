"""
=============================================================================
ROSA Knee AI — FastAPI Application Entry Point
=============================================================================
This is the main FastAPI application for the ROSA Knee AI backend.

Architecture:
    Frontend (React) ──HTTP──> FastAPI ──> Services ──> src/ library

How to run:
    cd ROSA_Knee_AI
    uvicorn backend.app.main:app --reload --port 8000

Then visit:
    http://localhost:8000/health          → health check
    http://localhost:8000/docs            → Swagger UI (interactive API docs)
    http://localhost:8000/api/v1/model/status → model availability

IMPORTANT:
    - No model training happens here
    - No fake predictions are returned
    - The model status honestly reports "not trained"
=============================================================================
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.exceptions import (
    RosaBaseException,
    rosa_exception_handler,
    generic_exception_handler,
)
from backend.app.schemas.responses import HealthResponse
from backend.app.api.v1.router import router as v1_router


# ---------------------------------------------------------------------------
# Application lifespan — startup & shutdown logic
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on application startup and shutdown.

    Startup: Ensure directories exist, log configuration.
    Shutdown: Clean up resources (future: unload model from GPU).
    """
    # --- STARTUP ---
    print("=" * 60)
    print(f"  {settings.app_title} v{settings.app_version}")
    print("=" * 60)
    print(f"  Data directory:   {settings.data_dir}")
    print(f"  Cases directory:  {settings.cases_dir}")
    print(f"  Model:            {settings.segmentation_model}")
    print(f"  Debug mode:       {settings.debug}")
    print("=" * 60)

    # Ensure required directories exist
    settings.cases_dir.mkdir(parents=True, exist_ok=True)

    yield  # Application runs here

    # --- SHUTDOWN ---
    print("ROSA Knee AI backend shutting down.")


# ---------------------------------------------------------------------------
# Create the FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Backend API for AI-based knee bone segmentation, "
        "3D reconstruction, and surgical planning for ROSA-assisted "
        "total knee arthroplasty.\n\n"
        "**Current status:** Development prototype. "
        "The segmentation model has not been trained yet."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS middleware — allow frontend dev server to make requests
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Register exception handlers — no raw tracebacks to users
# ---------------------------------------------------------------------------
app.add_exception_handler(RosaBaseException, rosa_exception_handler)
if not settings.debug:
    # In production, catch ALL unhandled exceptions with a safe message.
    # In debug mode, let FastAPI show the detailed error for development.
    app.add_exception_handler(Exception, generic_exception_handler)


# ---------------------------------------------------------------------------
# Health check endpoint — the simplest possible test
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns server health status. Use this to verify the backend is running.",
)
async def health_check():
    """Returns {"status": "ok"} if the server is running."""
    return HealthResponse(status="ok")


# ---------------------------------------------------------------------------
# Mount the v1 API router
# ---------------------------------------------------------------------------
app.include_router(
    v1_router,
    prefix=settings.api_v1_prefix,
    tags=["v1"],
)
