"""
=============================================================================
ROSA Knee AI — Processing Endpoints
=============================================================================
API endpoints for the processing pipeline:

    POST /cases/{id}/preprocess   → Run preprocessing
    POST /cases/{id}/segment      → Run segmentation (returns "unavailable")
    POST /cases/{id}/reconstruct  → Generate 3D meshes from masks
    GET  /cases/{id}/results      → Get all results and statuses
    GET  /cases/{id}/mesh/{name}  → Download an STL file

Pipeline:
    Upload → Validate → Preprocess → Segment → Reconstruct → Plan

Current status:
    ✓ Preprocess:   works with NIfTI/DICOM (not NPZ — no spacing)
    ✗ Segment:      returns "model unavailable"
    ✓ Reconstruct:  works IF a real segmentation mask (.npz) is supplied
    ✗ Plan:         returns "not available"
=============================================================================
"""

from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.services.case_manager import case_manager
from backend.app.core.exceptions import (
    CaseNotFoundError,
    ProcessingError,
    ModelUnavailableError,
)
from src.model.inference import get_segmentation_model
from src.reconstruction.mesh_generator import generate_all_meshes
from src.planning.planning_engine import planning_engine

router = APIRouter(prefix="/cases", tags=["Processing"])


# ---------------------------------------------------------------------------
# POST /cases/{id}/preprocess
# ---------------------------------------------------------------------------
@router.post(
    "/{case_id}/preprocess",
    summary="Run preprocessing on a case",
    description=(
        "Applies the preprocessing pipeline (resampling, windowing) to the "
        "uploaded CT data. Only works with formats that have spatial metadata "
        "(DICOM, NIfTI). NPZ files lack spacing information."
    ),
)
async def preprocess_case(case_id: str):
    """Run preprocessing on uploaded CT data."""
    case = case_manager.get_case(case_id)
    raw_dir = case_manager.get_raw_dir(case_id)
    preprocessed_dir = case_manager.get_preprocessed_dir(case_id)

    # Check if the case has been validated
    if case.status not in ("validated", "preprocessed"):
        raise ProcessingError(
            f"Case '{case_id}' has not been validated yet. "
            f"Current status: {case.status}"
        )

    # Try to load and preprocess using SimpleITK
    try:
        import SimpleITK as sitk
        from src.preprocessing.ct_preprocessing import resample_image

        # Find the medical image file
        image = None
        input_file = None

        # Check for NIfTI
        for ext in ("*.nii.gz", "*.nii", "*.nrrd", "*.mha"):
            files = list(raw_dir.glob(ext))
            if files:
                image = sitk.ReadImage(str(files[0]))
                input_file = files[0]
                break

        # Check for DICOM
        if image is None:
            reader = sitk.ImageSeriesReader()
            dcm_dir = raw_dir
            # Check subdirectories too
            for d in [raw_dir] + [x for x in raw_dir.iterdir() if x.is_dir()]:
                names = reader.GetGDCMSeriesFileNames(str(d))
                if names:
                    reader.SetFileNames(names)
                    image = reader.Execute()
                    input_file = d
                    break

        # Handle NPZ (no spatial metadata for resampling)
        if image is None:
            npz_files = list(raw_dir.glob("*.npz"))
            if npz_files:
                # NPZ doesn't have spacing, so we can't resample properly
                # Just copy it to preprocessed as-is and note the limitation
                with np.load(npz_files[0], allow_pickle=False) as data:
                    key = "x" if "x" in data else list(data.keys())[0]
                    arr = data[key]
                    out_path = preprocessed_dir / f"{npz_files[0].stem}_preprocessed.npz"
                    np.savez_compressed(out_path, x=arr)

                case_manager.update_case(
                    case_id,
                    status="preprocessed",
                    processing_status="preprocessed_no_spacing",
                )
                return {
                    "case_id": case_id,
                    "status": "preprocessed",
                    "message": (
                        "NPZ file copied to preprocessed directory. "
                        "Resampling was NOT applied because NPZ files do not "
                        "contain spacing metadata. Use DICOM or NIfTI for "
                        "proper spatial preprocessing."
                    ),
                }

        if image is None:
            raise ProcessingError(
                f"No loadable medical image found in case '{case_id}'."
            )

        # Resample to target spacing
        target_spacing = (0.5, 0.5, 0.5)
        resampled = resample_image(image, target_spacing=target_spacing)

        # Save preprocessed volume as NIfTI
        out_path = preprocessed_dir / "volume_preprocessed.nii.gz"
        sitk.WriteImage(resampled, str(out_path))

        # Update case metadata
        new_size = resampled.GetSize()
        new_spacing = resampled.GetSpacing()
        case_manager.update_case(
            case_id,
            status="preprocessed",
            processing_status="preprocessed",
            preprocessed_size=list(new_size),
            preprocessed_spacing=list(new_spacing),
        )

        return {
            "case_id": case_id,
            "status": "preprocessed",
            "original_spacing": list(image.GetSpacing()),
            "target_spacing": list(target_spacing),
            "new_size": list(new_size),
            "message": "Preprocessing complete. Volume resampled and saved.",
        }

    except ProcessingError:
        raise
    except Exception as e:
        raise ProcessingError(f"Preprocessing failed: {str(e)}")


# ---------------------------------------------------------------------------
# POST /cases/{id}/segment
# ---------------------------------------------------------------------------
@router.post(
    "/{case_id}/segment",
    summary="Run segmentation on a case",
    description=(
        "Runs the AI segmentation model on the preprocessed CT volume. "
        "Currently returns 'model unavailable' since no model has been trained."
    ),
)
async def segment_case(case_id: str):
    """
    Run segmentation. Currently returns 'model unavailable'.

    When nnU-Net is trained, this endpoint will:
        1. Load the preprocessed volume
        2. Run inference through the model
        3. Save the segmentation mask
        4. Return label information
    """
    # Verify the case exists
    case_manager.get_case(case_id)

    # Get the model (currently placeholder)
    model = get_segmentation_model("placeholder")

    # Check availability
    if not model.is_available():
        status = model.get_status()
        return {
            "case_id": case_id,
            "segmentation_status": "model_unavailable",
            "model_name": status["model_name"],
            "model_trained": status["trained"],
            "message": status["message"],
            "mask": None,
            "labels": None,
        }

    # Future: actual inference goes here
    result = model.predict(None)
    return {
        "case_id": case_id,
        "segmentation_status": result["status"],
        "message": result["message"],
    }


# ---------------------------------------------------------------------------
# POST /cases/{id}/reconstruct
# ---------------------------------------------------------------------------
@router.post(
    "/{case_id}/reconstruct",
    summary="Generate 3D meshes from segmentation masks",
    description=(
        "Runs Marching Cubes on the segmentation mask to produce STL files. "
        "Works with real masks from our dataset — does not require the AI model."
    ),
)
async def reconstruct_case(case_id: str):
    """
    Generate 3D mesh from segmentation mask.

    This works if:
        - A real .npz segmentation mask was uploaded to the case
        - Or a segmentation mask was produced by the model (future)
    """
    case = case_manager.get_case(case_id)
    raw_dir = case_manager.get_raw_dir(case_id)
    seg_dir = case_manager.get_segmentation_dir(case_id)
    meshes_dir = case_manager.get_meshes_dir(case_id)

    # Look for a segmentation mask
    mask = None

    # Check segmentation directory first (from model output)
    for f in seg_dir.glob("*.npz"):
        with np.load(f) as data:
            key = "x" if "x" in data else list(data.keys())[0]
            mask = data[key]
        break

    # If no model output, check raw directory for .npz masks
    if mask is None:
        for f in raw_dir.glob("*.npz"):
            with np.load(f) as data:
                key = "x" if "x" in data else list(data.keys())[0]
                arr = data[key]
                # Check if it looks like a segmentation mask
                unique = np.unique(arr)
                if len(unique) < 20 and arr.dtype in (np.uint8, np.int8, np.int16):
                    mask = arr
                    break

    if mask is None:
        return {
            "case_id": case_id,
            "status": "no_mask",
            "message": (
                "No segmentation mask found. Either upload a mask (.npz) "
                "or run segmentation first (model not yet available)."
            ),
            "meshes": [],
        }

    # Generate meshes for all labels
    # spacing is unknown for .npz — use (1,1,1) as placeholder
    results = generate_all_meshes(
        mask=mask,
        output_dir=meshes_dir,
        spacing=(1.0, 1.0, 1.0),
        step_size=2,  # Faster for preview
    )

    # Update case status
    successful = [r for r in results if r["status"] == "success"]
    case_manager.update_case(
        case_id,
        status="reconstructed" if successful else case.status,
    )

    return {
        "case_id": case_id,
        "status": "reconstructed" if successful else "reconstruction_failed",
        "num_meshes": len(successful),
        "meshes": results,
        "message": f"Generated {len(successful)} mesh(es).",
    }


# ---------------------------------------------------------------------------
# GET /cases/{id}/results
# ---------------------------------------------------------------------------
@router.get(
    "/{case_id}/results",
    summary="Get all results for a case",
    description="Returns segmentation, reconstruction, and planning status.",
)
async def get_results(case_id: str):
    """Get comprehensive results for a case."""
    case = case_manager.get_case(case_id)
    meshes_dir = case_manager.get_meshes_dir(case_id)

    # Check for STL files
    stl_files = list(meshes_dir.glob("*.stl"))
    mesh_list = [
        {"name": f.stem, "filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in stl_files
    ]

    # Get model status
    model = get_segmentation_model("placeholder")
    model_status = model.get_status()

    # Get planning status
    plan_status = planning_engine.get_status()

    return {
        "case_id": case_id,
        "case_status": case.status,
        "segmentation": {
            "status": "model_unavailable",
            "model": model_status,
        },
        "reconstruction": {
            "status": "available" if mesh_list else "not_available",
            "meshes": mesh_list,
        },
        "planning": plan_status,
    }


# ---------------------------------------------------------------------------
# GET /cases/{id}/mesh/{name}
# ---------------------------------------------------------------------------
@router.get(
    "/{case_id}/mesh/{mesh_name}",
    summary="Download an STL mesh file",
    description="Returns the STL file for a specific structure.",
)
async def get_mesh(case_id: str, mesh_name: str):
    """Download an STL mesh file."""
    meshes_dir = case_manager.get_meshes_dir(case_id)

    # Add .stl extension if not present
    if not mesh_name.endswith(".stl"):
        mesh_name += ".stl"

    stl_path = meshes_dir / mesh_name
    if not stl_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Mesh file '{mesh_name}' not found for case '{case_id}'."
        )

    return FileResponse(
        path=str(stl_path),
        media_type="application/octet-stream",
        filename=mesh_name,
    )
