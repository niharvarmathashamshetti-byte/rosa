#!/usr/bin/env python3
"""Validate the real OAI medical sample before using it in the app pipeline."""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    """Check that the OAI MRI/label pair is real medical data and shape-consistent."""
    image_path = PROJECT_ROOT / "data" / "raw" / "imagesTr" / "oaizib_005_0000.nii.gz"
    label_path = PROJECT_ROOT / "data" / "raw" / "labelsTr" / "oaizib_005.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"MRI sample not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label sample not found: {label_path}")

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    image_array = np.asarray(image.get_fdata(), dtype=np.float32)
    label_array = np.asarray(label.get_fdata(), dtype=np.float32)

    if image_array.shape != label_array.shape:
        raise ValueError(
            f"Shape mismatch: image {image_array.shape}, label {label_array.shape}. "
            "The MRI and segmentation must match exactly."
        )

    if not np.isfinite(image_array).all():
        raise ValueError(f"MRI sample contains non-finite values: {image_path}")
    if not np.isfinite(label_array).all():
        raise ValueError(f"Label sample contains non-finite values: {label_path}")

    unique_labels = np.unique(label_array)
    if unique_labels.size < 2:
        raise ValueError(f"Label sample does not look like a real multi-class mask: {unique_labels.tolist()}")

    print(f"Validated sample files: {image_path.name}, {label_path.name}")
    print(f"Image shape: {image_array.shape}, dtype: {image_array.dtype}")
    print(f"Intensity range: {float(image_array.min())} to {float(image_array.max())}")
    print(f"Label unique values: {unique_labels.tolist()}")
    print(f"Spacing: {tuple(float(v) for v in image.header.get_zooms()[:3])}")
    print("Status: real medical sample is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
