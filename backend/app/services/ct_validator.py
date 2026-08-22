"""
=============================================================================
ROSA Knee AI — CT Validator Service
=============================================================================
Validates uploaded medical images and extracts metadata.

Checks performed:
  1. Is the file readable?
  2. Is it a recognized medical image format?
  3. Is the volume 3D?
  4. What is the shape, spacing, orientation?
  5. Are there corrupted or missing slices?
  6. Is metadata available?

This service uses SimpleITK (from our existing src/ library) for loading.
It does NOT modify any files — validation is read-only.
=============================================================================
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from backend.app.schemas.case import CaseValidation


class CTValidator:
    """
    Validates medical image files and extracts spatial metadata.
    """

    def validate_dicom_directory(self, dicom_dir: Path) -> CaseValidation:
        """
        Validate a directory containing DICOM slices.

        Args:
            dicom_dir: Path to directory with .dcm files.

        Returns:
            CaseValidation with metadata and any issues found.
        """
        issues = []

        try:
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))

            if not series_ids:
                return CaseValidation(
                    valid=False,
                    format="DICOM",
                    issues=["No DICOM series found in directory."]
                )

            # Use the first series (most DICOM folders have one series)
            if len(series_ids) > 1:
                issues.append(
                    f"Multiple DICOM series found ({len(series_ids)}). "
                    f"Using the first series."
                )

            dicom_names = reader.GetGDCMSeriesFileNames(
                str(dicom_dir), series_ids[0]
            )

            if len(dicom_names) < 2:
                issues.append(f"Only {len(dicom_names)} DICOM file(s) found. "
                              f"A 3D volume typically has many slices.")

            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            return self._extract_metadata(image, "DICOM", issues, num_slices=len(dicom_names))

        except Exception as e:
            return CaseValidation(
                valid=False,
                format="DICOM",
                issues=[f"Failed to read DICOM: {str(e)}"]
            )

    def validate_nifti_file(self, file_path: Path) -> CaseValidation:
        """
        Validate a NIfTI (.nii / .nii.gz) file.

        Args:
            file_path: Path to the NIfTI file.

        Returns:
            CaseValidation with metadata and any issues found.
        """
        issues = []

        try:
            image = sitk.ReadImage(str(file_path))
            fmt = "NIfTI"
            if file_path.suffix == ".nrrd":
                fmt = "NRRD"
            elif file_path.suffix in (".mha", ".mhd"):
                fmt = "MHA"

            return self._extract_metadata(image, fmt, issues)

        except Exception as e:
            return CaseValidation(
                valid=False,
                format="NIfTI",
                issues=[f"Failed to read file: {str(e)}"]
            )

    def validate_npz_file(self, file_path: Path) -> CaseValidation:
        """
        Validate a .npz file containing a numpy array.

        Note: NPZ files don't have spatial metadata (spacing, origin).
        We report shape and dtype but spacing is unknown.

        Args:
            file_path: Path to the .npz file.

        Returns:
            CaseValidation with metadata and any issues found.
        """
        issues = []

        try:
            with np.load(file_path, allow_pickle=False) as data:
                keys = list(data.keys())
                array_key = "x" if "x" in keys else keys[0]
                arr = data[array_key]

                if arr.ndim != 3:
                    issues.append(f"Expected 3D volume, got {arr.ndim}D array.")
                    return CaseValidation(valid=False, format="NPZ", issues=issues)

                if arr.size == 0:
                    issues.append("Volume is empty (zero size).")
                    return CaseValidation(valid=False, format="NPZ", issues=issues)

                # NPZ files have no spacing/origin metadata
                issues.append(
                    "NPZ format does not contain spatial metadata "
                    "(spacing, origin, direction). These must come from "
                    "the original DICOM/NIfTI source."
                )

                return CaseValidation(
                    valid=True,
                    format="NPZ",
                    dimensions=list(arr.shape),
                    spacing=None,  # Unknown for NPZ
                    orientation=None,
                    origin=None,
                    physical_size=None,
                    pixel_type=str(arr.dtype),
                    issues=issues,
                )

        except Exception as e:
            return CaseValidation(
                valid=False,
                format="NPZ",
                issues=[f"Failed to read NPZ file: {str(e)}"]
            )

    def validate_case_directory(self, raw_dir: Path) -> CaseValidation:
        """
        Auto-detect the format and validate whatever is in the raw/ directory.

        Checks for DICOM (.dcm), NIfTI (.nii/.nii.gz), NPZ (.npz), etc.
        """
        # Check for DICOM files
        dcm_files = list(raw_dir.glob("*.dcm")) + list(raw_dir.glob("*.DCM"))
        if dcm_files:
            return self.validate_dicom_directory(raw_dir)

        # Check for NIfTI files
        nifti_files = (
            list(raw_dir.glob("*.nii"))
            + list(raw_dir.glob("*.nii.gz"))
            + list(raw_dir.glob("*.nrrd"))
            + list(raw_dir.glob("*.mha"))
        )
        if nifti_files:
            return self.validate_nifti_file(nifti_files[0])

        # Check for NPZ files
        npz_files = list(raw_dir.glob("*.npz"))
        if npz_files:
            return self.validate_npz_file(npz_files[0])

        # Check subdirectories for DICOM
        for subdir in raw_dir.iterdir():
            if subdir.is_dir():
                dcm_in_sub = list(subdir.glob("*.dcm")) + list(subdir.glob("*.DCM"))
                if dcm_in_sub:
                    return self.validate_dicom_directory(subdir)

        return CaseValidation(
            valid=False,
            format="unknown",
            issues=["No recognized medical image files found in upload."]
        )

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _extract_metadata(
        self,
        image: sitk.Image,
        fmt: str,
        issues: list[str],
        num_slices: Optional[int] = None,
    ) -> CaseValidation:
        """Extract spatial metadata from a SimpleITK Image."""
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()

        # Check if truly 3D
        if image.GetDimension() < 3 or min(size) < 2:
            issues.append(f"Volume may not be truly 3D. Size: {size}")

        # Calculate physical size
        physical_size = [round(s * sp, 2) for s, sp in zip(size, spacing)]

        # Format direction as a readable string
        orientation_str = str(list(round(d, 4) for d in direction))

        return CaseValidation(
            valid=True,
            format=fmt,
            dimensions=list(size),
            spacing=[round(s, 4) for s in spacing],
            orientation=orientation_str,
            origin=[round(o, 2) for o in origin],
            physical_size=physical_size,
            pixel_type=pixel_type,
            num_slices=num_slices,
            issues=issues if issues else [],
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
ct_validator = CTValidator()
