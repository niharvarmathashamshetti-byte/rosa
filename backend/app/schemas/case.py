"""
=============================================================================
ROSA Knee AI — Case Pydantic Schemas
=============================================================================
Data models for case management.

These define the shape of case data flowing through the API:
  - CaseCreate:   what the upload endpoint receives
  - CaseInfo:     what GET /cases/{id} returns
  - CaseSummary:  what GET /cases returns (list item)
  - CaseValidation: CT validation results
=============================================================================
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class CaseCreate(BaseModel):
    """Response returned after a successful case upload."""
    case_id: str = Field(description="Unique case identifier, e.g., 'case_000001'")
    status: str = Field(default="uploaded", description="Current case status")


class CaseValidation(BaseModel):
    """CT validation results for a case."""
    valid: bool = Field(description="Whether the CT data passed validation")
    format: str = Field(default="unknown", description="Detected format: DICOM, NIfTI, NPZ")
    dimensions: Optional[list[int]] = Field(default=None, description="Volume dimensions [x, y, z]")
    spacing: Optional[list[float]] = Field(default=None, description="Voxel spacing [x, y, z] in mm")
    orientation: Optional[str] = Field(default=None, description="Orientation string or matrix")
    origin: Optional[list[float]] = Field(default=None, description="Volume origin [x, y, z]")
    physical_size: Optional[list[float]] = Field(default=None, description="Physical size [x, y, z] in mm")
    pixel_type: Optional[str] = Field(default=None, description="Pixel/voxel data type")
    num_slices: Optional[int] = Field(default=None, description="Number of DICOM slices")
    issues: list[str] = Field(default_factory=list, description="List of validation issues found")


class CaseInfo(BaseModel):
    """Full case details returned by GET /cases/{id}."""
    case_id: str
    status: str = Field(description="uploaded, validated, preprocessed, segmented, reconstructed")
    created_at: str = Field(description="ISO 8601 timestamp")
    input_format: str = Field(default="unknown")

    # Volume metadata — populated after validation
    image_size: Optional[list[int]] = None
    voxel_spacing: Optional[list[float]] = None
    orientation: Optional[str] = None
    physical_size: Optional[list[float]] = None

    # Processing status
    processing_status: str = Field(default="not_started")
    model_status: str = Field(default="unavailable")

    # Validation details
    validation: Optional[CaseValidation] = None

    # Flags for what outputs exist
    has_preprocessed: bool = False
    has_segmentation: bool = False
    has_meshes: bool = False


class CaseSummary(BaseModel):
    """Compact case info for list endpoints."""
    case_id: str
    status: str
    created_at: str
    input_format: str
    image_size: Optional[list[int]] = None
    processing_status: str = "not_started"
