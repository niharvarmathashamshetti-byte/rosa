"""
=============================================================================
ROSA Knee AI — Case Management Endpoints
=============================================================================
API endpoints for creating, listing, viewing, and deleting cases.

Upload flow:
    1. User uploads a ZIP (containing DICOM slices) or a single NIfTI/NPZ file
    2. Backend extracts to data/cases/{case_id}/raw/
    3. CT validator checks the data and extracts metadata
    4. Metadata is stored in case.json
    5. Response includes case_id, status, and validation results

Supported upload formats:
    - .zip containing DICOM .dcm files (clinical workflow)
    - .nii or .nii.gz single file (research workflow)
    - .npz single file (our current dataset format)
    - .nrrd or .mha single file
=============================================================================
"""

import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.schemas.case import CaseCreate, CaseInfo, CaseSummary
from backend.app.services.case_manager import case_manager
from backend.app.services.ct_validator import ct_validator
from backend.app.core.config import settings
from backend.app.core.exceptions import InvalidUploadError, CaseNotFoundError

router = APIRouter(prefix="/cases", tags=["Cases"])


# ---------------------------------------------------------------------------
# POST /cases — Upload a new case
# ---------------------------------------------------------------------------
@router.post(
    "",
    response_model=CaseInfo,
    status_code=201,
    summary="Upload a new CT case",
    description=(
        "Upload a medical image file (.zip of DICOMs, .nii.gz, .npz). "
        "Creates a new case, extracts the data, validates it, "
        "and returns the case details with metadata."
    ),
)
async def create_case(file: UploadFile = File(...)):
    """
    Upload a CT study and create a new case.

    The uploaded file is saved to data/cases/{case_id}/raw/,
    then validated to extract spatial metadata.
    """
    # Validate the upload file extension
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    # Handle .nii.gz which has a double extension
    if filename.lower().endswith(".nii.gz"):
        suffix = ".nii.gz"

    allowed = [".zip", ".nii", ".nii.gz", ".nrrd", ".mha", ".npz"]
    if suffix not in allowed:
        raise InvalidUploadError(
            f"Unsupported file format: '{suffix}'. "
            f"Allowed formats: {', '.join(allowed)}"
        )

    # Check file size (read content-length header if available)
    # For streaming uploads, we check as we write
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    # Create the case
    case_id = case_manager.create_case()
    raw_dir = case_manager.get_raw_dir(case_id)

    try:
        # Save the uploaded file
        save_path = raw_dir / filename
        total_bytes = 0

        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(8 * 1024 * 1024)  # 8MB chunks
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    # Clean up and raise error
                    case_manager.delete_case(case_id)
                    raise InvalidUploadError(
                        f"File exceeds maximum upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    )
                f.write(chunk)

        # If it's a ZIP, extract it
        if suffix == ".zip":
            try:
                with zipfile.ZipFile(save_path, "r") as zf:
                    zf.extractall(raw_dir)
                # Remove the zip after extraction
                save_path.unlink()
            except zipfile.BadZipFile:
                case_manager.delete_case(case_id)
                raise InvalidUploadError("The uploaded ZIP file is corrupted.")

        # Determine format
        input_format = {
            ".zip": "DICOM",
            ".nii": "NIfTI",
            ".nii.gz": "NIfTI",
            ".nrrd": "NRRD",
            ".mha": "MHA",
            ".npz": "NPZ",
        }.get(suffix, "unknown")

        # Validate the CT data
        validation = ct_validator.validate_case_directory(raw_dir)

        # Update case metadata with validation results
        update_data = {
            "status": "validated" if validation.valid else "validation_failed",
            "input_format": validation.format or input_format,
            "validation": validation.model_dump(),
        }

        if validation.dimensions:
            update_data["image_size"] = validation.dimensions
        if validation.spacing:
            update_data["voxel_spacing"] = validation.spacing
        if validation.orientation:
            update_data["orientation"] = validation.orientation
        if validation.physical_size:
            update_data["physical_size"] = validation.physical_size

        case_manager.update_case(case_id, **update_data)

        # Return full case info
        return case_manager.get_case(case_id)

    except (InvalidUploadError, CaseNotFoundError):
        raise  # Re-raise our custom exceptions
    except Exception as e:
        # Clean up on unexpected errors
        try:
            case_manager.delete_case(case_id)
        except Exception:
            pass
        raise InvalidUploadError(f"Upload processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /cases — List all cases
# ---------------------------------------------------------------------------
@router.get(
    "",
    response_model=list[CaseSummary],
    summary="List all cases",
    description="Returns a list of all uploaded cases with summary information.",
)
async def list_cases():
    """List all cases in the system."""
    return case_manager.list_cases()


# ---------------------------------------------------------------------------
# GET /cases/{case_id} — Get case details
# ---------------------------------------------------------------------------
@router.get(
    "/{case_id}",
    response_model=CaseInfo,
    summary="Get case details",
    description="Returns full details for a specific case including validation results.",
)
async def get_case(case_id: str):
    """Get detailed information about a specific case."""
    return case_manager.get_case(case_id)


# ---------------------------------------------------------------------------
# DELETE /cases/{case_id} — Delete a case
# ---------------------------------------------------------------------------
@router.delete(
    "/{case_id}",
    summary="Delete a case",
    description="Permanently deletes a case and all associated data.",
)
async def delete_case(case_id: str):
    """Delete a case and all its data (raw, preprocessed, meshes, etc.)."""
    case_manager.delete_case(case_id)
    return {"message": f"Case '{case_id}' deleted successfully."}
