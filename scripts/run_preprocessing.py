#!/usr/bin/env python3
"""
=============================================================================
run_preprocessing.py — Preprocessing Pipeline Script
=============================================================================
Reads preprocessing configuration and applies it to dataset files.

STATUS: SKELETON — Do not use until data has been verified.
        We first need to confirm whether our .npz data contains CT intensity
        volumes or segmentation labels before applying any preprocessing.

Usage (after data verification):
    python scripts/run_preprocessing.py --config configs/preprocessing.yaml --data-dir data/raw
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point for the preprocessing pipeline."""

    # -----------------------------------------------------------------------
    # Parse command-line arguments
    # -----------------------------------------------------------------------
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
        help="Path for preprocessed output (default: from config)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("ROSA Knee AI — Preprocessing Pipeline")
    print("=" * 60)
    print(f"\nConfig: {config_path}")
    print(f"Target spacing: {config['resampling']['target_spacing']}")
    print(f"Bone window: center={config['windowing']['bone']['window_center']}, "
          f"width={config['windowing']['bone']['window_width']}")

    # -----------------------------------------------------------------------
    # STOP: Verify data type first
    # -----------------------------------------------------------------------
    print("\n" + "!" * 60)
    print("WARNING: Preprocessing is not yet enabled.")
    print()
    print("Before running preprocessing, you MUST verify:")
    print("  1. Whether the data contains CT intensity or segmentation labels")
    print("  2. What spatial metadata is available")
    print("  3. What preprocessing steps are appropriate")
    print()
    print("Run the exploration notebook first:")
    print("  notebooks/01_dataset_exploration.ipynb")
    print()
    print("Or run the dataset inspection script:")
    print("  python scripts/inspect_dataset.py --data-dir data/raw")
    print("!" * 60)

    # TODO: Implement preprocessing pipeline after data verification
    # Steps will include:
    #   1. Load each case
    #   2. Apply appropriate preprocessing based on data type
    #   3. Save preprocessed output with metadata
    #   4. Log all parameters for reproducibility


if __name__ == "__main__":
    main()
