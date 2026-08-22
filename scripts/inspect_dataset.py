#!/usr/bin/env python3
"""
=============================================================================
inspect_dataset.py — Dataset Inspection Script
=============================================================================
Scans a dataset directory, checks every file, and produces a CSV report.

Usage:
    python scripts/inspect_dataset.py --data-dir data/raw
    python scripts/inspect_dataset.py --data-dir data/raw --output outputs/reports/my_report.csv

What it does:
    1. Recursively scans the data directory for medical image files
    2. Loads each file and extracts metadata (shape, dtype, values, etc.)
    3. Classifies each volume as CT intensity or segmentation labels
    4. Checks for quality issues (corrupted, zero-sized, NaN values, etc.)
    5. Saves a CSV report with per-case information
    6. Prints a formatted summary to the console
=============================================================================
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Add project root to Python path so we can import from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quality_control.dataset_checker import (
    scan_dataset,
    generate_dataset_report,
    print_dataset_summary,
)


def main():
    """Main entry point for the dataset inspection script."""

    # -----------------------------------------------------------------------
    # Parse command-line arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Inspect a medical imaging dataset and generate a quality report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/inspect_dataset.py --data-dir data/raw
    python scripts/inspect_dataset.py --data-dir data/raw --output reports/custom_report.csv
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset directory (e.g., data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output CSV report (default: outputs/reports/dataset_report.csv)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------------
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    if not data_dir.exists():
        print(f"\n[ERROR] Data directory not found: {data_dir}")
        print("Please check the path and try again.")
        print("Make sure your dataset is placed in data/raw/")
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / "reports" / "dataset_report.csv"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    # -----------------------------------------------------------------------
    # Step 1: Scan for files
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROSA Knee AI — Dataset Inspection")
    print("=" * 60)
    print(f"\nScanning: {data_dir}")

    scan_results = scan_dataset(data_dir)

    # -----------------------------------------------------------------------
    # Step 2: Generate full report
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("Generating detailed report...")
    print(f"{'─' * 60}\n")

    report_df = generate_dataset_report(data_dir, output_path=output_path)

    # -----------------------------------------------------------------------
    # Step 3: Print summary
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print_dataset_summary(report_df)

    # -----------------------------------------------------------------------
    # Step 4: Report location
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print(f"Full report saved to: {output_path}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
