"""
01_dataset_verification.py
============================
STEP 2 + 3 + 4 of the training process (spec Section 5 and 6).

Performs ALL 19 checks from spec Section 5 before anything is allowed
to train:

 1. Verify the four (or two, if test not yet provided) paths exist
 2. Verify they are readable
 3. Extract zips (if given as zips) to DATA_DIR — originals untouched
 4-7. Locate imagesTr / labelsTr / imagesTs / labelsTs
 8-9. List all image / label files
 10. Match cases by ID
 11. Detect missing images
 12. Detect missing labels
 13. Detect duplicate case IDs
 14. Verify image/label voxel-grid shape match
 15. Verify spatial metadata (spacing, affine) match
 16. Verify NIfTI readability
 17. Verify labels contain only expected class IDs (spec Section 6)
 18. Detect empty foreground masks
 19. Print + save a complete dataset integrity report

If ANY critical check fails, this script exits non-zero and NO
downstream script should be run. Nothing is auto-"fixed".
"""

import sys
import zipfile
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

from config import (
    IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS,
    DATA_DIR, FINAL_RESULTS_DIR, INTEGRITY_REPORT_JSON,
    IMAGE_CHANNEL_SUFFIX, NIFTI_EXTENSIONS, EXPECTED_LABEL_SET, LABEL_MAP,
)
from utils import (
    log, fail, require, ensure_dirs, load_nifti,
    case_id_from_image_filename, case_id_from_label_filename,
    file_sha1_short, save_json,
)


def resolve_input(path: Path, extract_to: Path, label: str) -> Path:
    """Accepts either a directory or a .zip. Returns a directory path
    containing the actual .nii.gz files. Never modifies the original."""
    require(path is not None, f"{label} path is not set in config.py")
    path = Path(path)
    require(path.exists(), f"{label} path does not exist: {path}")

    if path.is_dir():
        log(f"{label}: using existing directory {path}")
        return path

    if path.suffix == ".zip":
        require(zipfile.is_zipfile(path), f"{label} is named .zip but is not a valid zip: {path}")
        target = extract_to / path.stem
        if target.exists() and any(target.iterdir()):
            log(f"{label}: already extracted at {target}, skipping re-extraction")
        else:
            target.mkdir(parents=True, exist_ok=True)
            log(f"{label}: extracting {path} -> {target}")
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(target)
            except Exception as e:
                fail(f"Failed to extract {label} zip {path}: {e}")
        return target

    fail(f"{label} path is neither a directory nor a .zip file: {path}")


def list_nifti_files(directory: Path):
    files = []
    for ext in NIFTI_EXTENSIONS:
        files.extend(sorted(directory.rglob(f"*{ext}")))
    # de-duplicate (rglob with .nii and .nii.gz could double count if both exist)
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def build_case_map(files, is_label: bool):
    """Returns dict case_id -> [paths] (list, so duplicates are visible)."""
    mapping = defaultdict(list)
    unmatched = []
    for f in files:
        if is_label:
            cid = case_id_from_label_filename(f.name, NIFTI_EXTENSIONS)
        else:
            cid = case_id_from_image_filename(f.name, IMAGE_CHANNEL_SUFFIX, NIFTI_EXTENSIONS)
        if cid is None:
            unmatched.append(str(f))
        else:
            mapping[cid].append(f)
    return mapping, unmatched


def verify_pair(case_id, img_path, lbl_path, report):
    """Checks 14-18 for a single image/label pair."""
    pair_report = {"case_id": case_id, "issues": []}

    img_data, img_affine, img_header = load_nifti(img_path)
    lbl_data, lbl_affine, lbl_header = load_nifti(lbl_path)

    # 14: shape match
    if img_data.shape != lbl_data.shape:
        pair_report["issues"].append(
            f"SHAPE MISMATCH: image {img_data.shape} vs label {lbl_data.shape}"
        )

    # 15: spatial metadata (affine / spacing) match
    if not np.allclose(img_affine, lbl_affine, atol=1e-3):
        pair_report["issues"].append("AFFINE MISMATCH between image and label")

    img_spacing = img_header.get_zooms()
    lbl_spacing = lbl_header.get_zooms()
    if not np.allclose(img_spacing, lbl_spacing, atol=1e-3):
        pair_report["issues"].append(
            f"SPACING MISMATCH: image {img_spacing} vs label {lbl_spacing}"
        )

    # 17: label values must be a subset of expected set
    unique_vals = set(np.unique(lbl_data).astype(int).tolist())
    unexpected = unique_vals - EXPECTED_LABEL_SET
    if unexpected:
        pair_report["issues"].append(
            f"UNEXPECTED LABEL VALUES: {sorted(unexpected)} "
            f"(expected only {sorted(EXPECTED_LABEL_SET)})"
        )
    pair_report["unique_label_values"] = sorted(unique_vals)

    # 18: empty foreground mask
    foreground_voxels = int(np.sum(lbl_data > 0))
    pair_report["foreground_voxel_count"] = foreground_voxels
    if foreground_voxels == 0:
        pair_report["issues"].append("EMPTY FOREGROUND MASK (all background)")

    pair_report["image_shape"] = list(img_data.shape)
    pair_report["voxel_spacing"] = [round(float(s), 4) for s in img_spacing]

    return pair_report


def verify_split(images_dir, labels_dir, split_name, report):
    log(f"--- Verifying split: {split_name} ---")
    image_files = list_nifti_files(images_dir)
    label_files = list_nifti_files(labels_dir)
    log(f"{split_name}: found {len(image_files)} image files, {len(label_files)} label files")

    img_map, img_unmatched = build_case_map(image_files, is_label=False)
    lbl_map, lbl_unmatched = build_case_map(label_files, is_label=True)

    if img_unmatched:
        log(f"{split_name}: {len(img_unmatched)} image filenames did not match "
            f"expected pattern '<case>{IMAGE_CHANNEL_SUFFIX}.nii.gz': "
            f"{img_unmatched[:5]}{' ...' if len(img_unmatched) > 5 else ''}",
            level="WARNING")
    if lbl_unmatched:
        log(f"{split_name}: {len(lbl_unmatched)} label filenames did not match "
            f"expected pattern '<case>.nii.gz': "
            f"{lbl_unmatched[:5]}{' ...' if len(lbl_unmatched) > 5 else ''}",
            level="WARNING")

    # 13: duplicate case IDs (more than one file mapped to same case id)
    dup_images = {cid: paths for cid, paths in img_map.items() if len(paths) > 1}
    dup_labels = {cid: paths for cid, paths in lbl_map.items() if len(paths) > 1}
    if dup_images:
        fail(f"{split_name}: DUPLICATE case IDs in images: "
             f"{ {k: [str(p) for p in v] for k, v in dup_images.items()} }")
    if dup_labels:
        fail(f"{split_name}: DUPLICATE case IDs in labels: "
             f"{ {k: [str(p) for p in v] for k, v in dup_labels.items()} }")

    image_ids = set(img_map.keys())
    label_ids = set(lbl_map.keys())

    # 11 / 12: missing images / labels
    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    if missing_labels:
        log(f"{split_name}: images with NO matching label: {missing_labels}", level="ERROR")
    if missing_images:
        log(f"{split_name}: labels with NO matching image: {missing_images}", level="ERROR")

    matched_ids = sorted(image_ids & label_ids)
    log(f"{split_name}: {len(matched_ids)} matched image/label pairs")

    if missing_labels or missing_images:
        fail(f"{split_name}: dataset is INCOMPLETE (see missing images/labels above). "
             f"Fix the dataset before proceeding — this pipeline will not train on "
             f"a partially-labeled dataset.")

    require(len(matched_ids) > 0, f"{split_name}: zero matched cases found — check paths and naming convention.")

    # 14-18 per pair
    pair_reports = []
    all_unique_labels_seen = set()
    empty_mask_cases = []
    mismatch_cases = []
    for cid in matched_ids:
        pr = verify_pair(cid, img_map[cid][0], lbl_map[cid][0], report)
        pair_reports.append(pr)
        all_unique_labels_seen.update(pr["unique_label_values"])
        if "EMPTY FOREGROUND MASK (all background)" in pr["issues"]:
            empty_mask_cases.append(cid)
        if any("MISMATCH" in issue for issue in pr["issues"]):
            mismatch_cases.append(cid)

    unexpected_overall = all_unique_labels_seen - EXPECTED_LABEL_SET
    if unexpected_overall:
        fail(f"{split_name}: dataset contains label values {sorted(unexpected_overall)} "
             f"not in the expected mapping {LABEL_MAP}. STOPPING per spec Section 6 — "
             f"the label mapping will NOT be silently changed. Inspect these cases "
             f"manually: "
             f"{[p['case_id'] for p in pair_reports if set(p['unique_label_values']) - EXPECTED_LABEL_SET]}")

    if mismatch_cases:
        fail(f"{split_name}: shape/spacing/affine mismatches found in cases: "
             f"{mismatch_cases}. STOPPING — these pairs cannot be trusted for training.")

    if empty_mask_cases:
        log(f"{split_name}: {len(empty_mask_cases)} case(s) have EMPTY foreground "
            f"masks: {empty_mask_cases}. This is not automatically fatal (a knee "
            f"could plausibly have zero segmented voxels only if something is "
            f"wrong with that specific label file), but you should manually "
            f"inspect these before training.", level="WARNING")

    shapes = [tuple(p["image_shape"]) for p in pair_reports]
    spacings = np.array([p["voxel_spacing"] for p in pair_reports])
    split_summary = {
        "num_cases": len(matched_ids),
        "case_ids": matched_ids,
        "unique_shapes_seen": sorted(set(shapes)),
        "spacing_min": spacings.min(axis=0).tolist(),
        "spacing_max": spacings.max(axis=0).tolist(),
        "spacing_mean": spacings.mean(axis=0).tolist(),
        "empty_mask_cases": empty_mask_cases,
        "all_unique_label_values_seen": sorted(all_unique_labels_seen),
        "per_case": pair_reports,
    }
    report[split_name] = split_summary
    log(f"{split_name}: PASSED all critical integrity checks "
        f"({len(matched_ids)} cases, label values seen: {sorted(all_unique_labels_seen)})")
    return split_summary


def main():
    ensure_dirs([DATA_DIR, FINAL_RESULTS_DIR])
    log("=" * 70)
    log("STEP 2/3/4: DATASET EXTRACTION + INTEGRITY VERIFICATION")
    log("=" * 70)

    report = {"label_map": LABEL_MAP}

    images_tr_dir = resolve_input(IMAGES_TR, DATA_DIR, "imagesTr")
    labels_tr_dir = resolve_input(LABELS_TR, DATA_DIR, "labelsTr")
    verify_split(images_tr_dir, labels_tr_dir, "train", report)

    if IMAGES_TS is not None and LABELS_TS is not None:
        images_ts_dir = resolve_input(IMAGES_TS, DATA_DIR, "imagesTs")
        labels_ts_dir = resolve_input(LABELS_TS, DATA_DIR, "labelsTs")
        verify_split(images_ts_dir, labels_ts_dir, "test", report)

        # Cross-split case-ID collision check (spec Section 13)
        train_ids = set(report["train"]["case_ids"])
        test_ids = set(report["test"]["case_ids"])
        overlap = train_ids & test_ids
        if overlap:
            fail(f"CASE ID OVERLAP between train and test sets: {sorted(overlap)}. "
                 f"This is a critical data-leakage risk. STOPPING.")
        log(f"No case-ID overlap between train ({len(train_ids)}) and "
            f"test ({len(test_ids)}) sets — confirmed clean split.")
    else:
        log("IMAGES_TS / LABELS_TS not set in config.py — test-set verification "
            "and the final held-out evaluation (spec Step 17) will be SKIPPED "
            "until you provide those paths. This is expected, not an error.",
            level="WARNING")
        report["test"] = None

    save_json(report, INTEGRITY_REPORT_JSON)
    log(f"Full integrity report saved to {INTEGRITY_REPORT_JSON}")
    log("=" * 70)
    log("DATASET INTEGRITY VERIFICATION: PASSED")
    log("=" * 70)


if __name__ == "__main__":
    main()
