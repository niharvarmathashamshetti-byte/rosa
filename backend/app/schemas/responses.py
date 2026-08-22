"""
=============================================================================
ROSA Knee AI — Pydantic Response Schemas
=============================================================================
These models define the shape of every JSON response the API returns.

Using Pydantic models ensures:
  1. Consistent response structure across all endpoints
  2. Automatic OpenAPI/Swagger documentation
  3. Type validation before sending data to the frontend
=============================================================================
"""

from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = Field(default="ok", description="Server health status")


class ErrorResponse(BaseModel):
    """Standard error response returned by all error handlers."""
    error: bool = Field(default=True)
    message: str = Field(description="Human-readable error message")
    detail: str = Field(default="", description="Additional context")


class ModelStatusResponse(BaseModel):
    """Response for GET /api/v1/model/status."""
    model_name: str = Field(description="Name of the segmentation model")
    status: str = Field(description="'available', 'unavailable', or 'loading'")
    message: str = Field(description="Human-readable status message")
    trained: bool = Field(default=False, description="Whether the model has been trained")
