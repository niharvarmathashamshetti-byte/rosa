"""
=============================================================================
ROSA Knee AI — CT Slice Viewer Endpoints
=============================================================================
Endpoints to serve 2D slice images as PNG for the web viewer.

Medical Imaging Context:
  - Axis 0: Axial (Z)
  - Axis 1: Coronal (Y)
  - Axis 2: Sagittal (X)

Handles:
  1. Real 3D CT Volumes (NIfTI, DICOM) -> converts slice to PNG with windowing
  2. 3D Segmentation masks (NPZ) -> converts slice to color-mapped PNG
  3. Cases without volume data -> returns 404 / clear error message
=============================================================================
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.app.services.case_manager import case_manager

router = APIRouter(prefix="/cases", tags=["Viewer"])


def _extract_slice_array(raw_dir: Path, axis: int, index: int) -> tuple[np.ndarray, str]:
    """
    Finds and loads the 3D volume in raw_dir, then extracts a 2D slice.
    Returns (slice_2d, data_type) where data_type is 'ct' or 'mask'.
    """
    # 1. Try SimpleITK formats (NIfTI, NRRD, MHA, DICOM)
    for ext in ("*.nii.gz", "*.nii", "*.nrrd", "*.mha"):
        files = list(raw_dir.glob(ext))
        if files:
            img = sitk.ReadImage(str(files[0]))
            arr = sitk.GetArrayFromImage(img)  # Shape in numpy is (Z, Y, X)
            # sitk: GetArrayFromImage gives (dim_z, dim_y, dim_x)
            return _slice_numpy(arr, axis, index), "ct"

    # Check DICOM
    reader = sitk.ImageSeriesReader()
    for d in [raw_dir] + [x for x in raw_dir.iterdir() if x.is_dir()]:
        names = reader.GetGDCMSeriesFileNames(str(d))
        if names:
            reader.SetFileNames(names)
            img = reader.Execute()
            arr = sitk.GetArrayFromImage(img)
            return _slice_numpy(arr, axis, index), "ct"

    # 2. Try NPZ
    npz_files = list(raw_dir.glob("*.npz"))
    if npz_files:
        with np.load(npz_files[0], allow_pickle=False) as data:
            key = "x" if "x" in data else list(data.keys())[0]
            arr = data[key]
            # Detect if mask or CT
            unique_count = len(np.unique(arr))
            dtype = "mask" if unique_count < 20 else "ct"
            return _slice_numpy(arr, axis, index), dtype

    raise FileNotFoundError("No viewable volume data found in case directory.")


def _slice_numpy(arr: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Safely slices a 3D numpy array along axis 0, 1, or 2."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    # Clamp index
    max_idx = arr.shape[axis] - 1
    idx = max(0, min(index, max_idx))

    if axis == 0:
        return arr[idx, :, :]
    elif axis == 1:
        return arr[:, idx, :]
    elif axis == 2:
        return arr[:, :, idx]
    else:
        raise ValueError("Axis must be 0 (Axial), 1 (Coronal), or 2 (Sagittal).")


@router.get(
    "/{case_id}/slices/{axis}/{index}",
    summary="Get 2D slice as PNG image",
    description="Renders a specific slice from the volume as a PNG image with windowing/colormap.",
)
async def get_slice_image(
    case_id: str,
    axis: int,
    index: int,
    window_center: Optional[float] = Query(400, description="Window center (Level) in HU"),
    window_width: Optional[float] = Query(1500, description="Window width in HU"),
):
    """
    Renders and returns a 2D slice as a PNG.
    """
    case = case_manager.get_case(case_id)
    raw_dir = case_manager.get_raw_dir(case_id)

    try:
        slice_2d, data_type = _extract_slice_array(raw_dir, axis, index)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No volume found for case '{case_id}'. CT viewer data not available.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render slice: {str(e)}")

    # Render slice to PNG buffer
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if data_type == "ct":
        # Apply CT bone windowing
        min_hu = window_center - (window_width / 2.0)
        max_hu = window_center + (window_width / 2.0)
        clipped = np.clip(slice_2d, min_hu, max_hu)
        ax.imshow(clipped, cmap="gray", vmin=min_hu, vmax=max_hu)
    else:
        # Categorical mask colormap
        ax.imshow(slice_2d, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/png")


@router.get(
    "/{case_id}/volume-info",
    summary="Get volume dimensions for viewer slider setup",
)
async def get_volume_info(case_id: str):
    """Returns shape along all 3 axes for viewer configuration."""
    case = case_manager.get_case(case_id)
    raw_dir = case_manager.get_raw_dir(case_id)

    try:
        # Sample slice on axis 0 to verify loading & get full shape
        for ext in ("*.nii.gz", "*.nii", "*.nrrd", "*.mha"):
            files = list(raw_dir.glob(ext))
            if files:
                img = sitk.ReadImage(str(files[0]))
                size = img.GetSize()  # (X, Y, Z) in sitk
                return {
                    "available": True,
                    "data_type": "ct",
                    "shape": [size[2], size[1], size[0]],  # [Z, Y, X]
                    "axial_slices": size[2],
                    "coronal_slices": size[1],
                    "sagittal_slices": size[0],
                }

        npz_files = list(raw_dir.glob("*.npz"))
        if npz_files:
            with np.load(npz_files[0], allow_pickle=False) as data:
                key = "x" if "x" in data else list(data.keys())[0]
                shape = list(data[key].shape)
                return {
                    "available": True,
                    "data_type": "mask",
                    "shape": shape,
                    "axial_slices": shape[0],
                    "coronal_slices": shape[1],
                    "sagittal_slices": shape[2],
                }

        return {
            "available": False,
            "message": "CT viewer data not available for this case.",
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Error retrieving volume info: {str(e)}",
        }
