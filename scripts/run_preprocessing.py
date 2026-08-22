#!/usr/bin/env python3
"""
=============================================================================
run_preprocessing.py — Preprocessing Pipeline Script
=============================================================================
Resamples medical images in a dataset directory to the configured target spacing
and saves the results as NIfTI files.

This script is intentionally conservative: it only processes files that look like
medical images (NIfTI / NRRD / MHA / MHD). It does not attempt to invent label
meanings or alter segmentation masks without metadata.
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk
import yaml

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ct_preprocessing import resample_image


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _iter_input_files(data_dir: Path):
    patterns = ("*.nii.gz", "*.nii", "*.nrrd", "*.mha", "*.mhd")
    files = []
    for pattern in patterns:
        files.extend(sorted(data_dir.glob(pattern)))
    return files


def _preprocessed_output_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7] + "_preprocessed.nii.gz"
    if name.endswith(".nii"):
        return name[:-4] + "_preprocessed.nii.gz"
    if name.endswith(".nrrd"):
        return name[:-5] + "_preprocessed.nii.gz"
    if name.endswith(".mha"):
        return name[:-4] + "_preprocessed.nii.gz"
    if name.endswith(".mhd"):
        return name[:-4] + "_preprocessed.nii.gz"
    return name + "_preprocessed.nii.gz"


def main():
    """Main entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the preprocessing pipeline on a medical imaging dataset.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing.yaml",
        help="Path to preprocessing config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path for preprocessed output (default: from config or a sibling folder)",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    if not config_path.exists():
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    target_spacing = tuple(
        float(v) for v in config.get("resampling", {}).get("target_spacing", [0.5, 0.5, 0.5])
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        config.get("output", {}).get("save_dir", "data/processed")
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = _iter_input_files(data_dir)
    if not input_files:
        print(f"\n[ERROR] No medical image files found in {data_dir}")
        print("Supported files: .nii, .nii.gz, .nrrd, .mha, .mhd")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("ROSA Knee AI — Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfig: {config_path}")
    print(f"Target spacing: {target_spacing}")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}\n")

    processed = []
    for image_path in input_files:
        image = sitk.ReadImage(str(image_path))
        resampled = resample_image(image, target_spacing=target_spacing)

        output_path = output_dir / _preprocessed_output_name(image_path)
        sitk.WriteImage(resampled, str(output_path))
        processed.append(str(output_path))
        print(f"Processed: {image_path.name} -> {output_path.name}")

    print(f"\nFinished: {len(processed)} file(s) processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
