#!/usr/bin/env python3
"""
Run one complete 3D reconstruction example using the available knee MRI + label pair.

This script intentionally uses the repo's real NIfTI data, not synthetic masks,
so it demonstrates how the project can generate meshes from actual medical data.

It does not claim anatomical label meanings unless they are verified.
"""

from pathlib import Path
import sys

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.mesh_generator import generate_all_meshes


def main():
    image_path = PROJECT_ROOT / "data" / "raw" / "imagesTr" / "oaizib_005_0000.nii.gz"
    label_path = PROJECT_ROOT / "data" / "raw" / "labelsTr" / "oaizib_005.nii.gz"
    output_dir = PROJECT_ROOT / "outputs" / "meshes" / "demo_oaizib_005"

    if not image_path.exists():
        raise FileNotFoundError(f"MRI file not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    img = nib.load(str(image_path))
    label_img = nib.load(str(label_path))

    img_arr = np.asarray(img.get_fdata())
    label_arr = np.asarray(label_img.get_fdata())

    if img_arr.shape != label_arr.shape:
        raise ValueError(
            f"Shape mismatch: image {img_arr.shape}, label {label_arr.shape}. "
            "The MRI and segmentation must be same geometry."
        )

    print("== Example dataset ==")
    print(f"MRI path: {image_path}")
    print(f"Label path: {label_path}")
    print(f"MRI shape: {img_arr.shape}")
    print(f"Label unique values: {np.unique(label_arr)[:20].tolist()}")
    print(f"Spacing: {img.header.get_zooms()}")

    # Use integer mask for mesh generation.
    mask = np.asarray(label_arr, dtype=np.uint8)
    spacing = tuple(float(v) for v in img.header.get_zooms()[:3])

    print("\n== Mesh reconstruction ==")
    results = generate_all_meshes(
        mask=mask,
        output_dir=output_dir,
        spacing=spacing,
        step_size=2,
    )

    print(f"Generated {len(results)} mesh result(s)")
    for result in results:
        print(result)

    stl_files = sorted(output_dir.glob("*.stl"))
    print(f"\nSTL files saved: {len(stl_files)}")
    for f in stl_files:
        print(f" - {f.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
