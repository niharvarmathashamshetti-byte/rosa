"""
02_convert_to_nnunet_format.py
================================
STEP 5 + 6 of the training process (spec Section 14).

Requires 01_dataset_verification.py to have PASSED first (it re-checks
the saved integrity report and refuses to run otherwise).

Builds the nnU-Net v2 raw dataset layout:

    nnUNet_raw/
      DatasetXXX_KneeOAIZIB/
        imagesTr/  <case>_0000.nii.gz
        labelsTr/  <case>.nii.gz
        imagesTs/  <case>_0000.nii.gz   (only if test paths were provided)
        dataset.json

Files are COPIED (not moved/linked) from the verified source directories
so your original downloads remain untouched.
"""

import json
import shutil
import sys
from pathlib import Path

from config import (
    NNUNET_RAW_DIR, NNUNET_FULL_NAME, DATA_DIR,
    IMAGES_TR, LABELS_TR, IMAGES_TS, LABELS_TS,
    LABEL_MAP, IMAGE_CHANNEL_SUFFIX, NIFTI_EXTENSIONS,
    INTEGRITY_REPORT_JSON,
)
from utils import log, fail, require, ensure_dirs
from importlib import import_module

verify_mod = import_module("01_dataset_verification")


def load_verified_report():
    if not INTEGRITY_REPORT_JSON.exists():
        fail("No integrity report found. Run 01_dataset_verification.py "
             "successfully before this script.")
    with open(INTEGRITY_REPORT_JSON) as f:
        return json.load(f)


def copy_split(case_ids, src_images_dir, src_labels_dir, dst_images_dir,
                dst_labels_dir, has_labels=True):
    ensure_dirs([dst_images_dir])
    if has_labels:
        ensure_dirs([dst_labels_dir])

    img_files = verify_mod.list_nifti_files(Path(src_images_dir))
    lbl_files = verify_mod.list_nifti_files(Path(src_labels_dir)) if has_labels else []

    img_map, _ = verify_mod.build_case_map(img_files, is_label=False)
    lbl_map, _ = verify_mod.build_case_map(lbl_files, is_label=True) if has_labels else ({}, [])

    for cid in case_ids:
        src_img = img_map[cid][0]
        dst_img = Path(dst_images_dir) / f"{cid}{IMAGE_CHANNEL_SUFFIX}.nii.gz"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        if has_labels:
            src_lbl = lbl_map[cid][0]
            dst_lbl = Path(dst_labels_dir) / f"{cid}.nii.gz"
            if not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    log(f"Copied {len(case_ids)} cases -> {dst_images_dir}"
        + (f" and {dst_labels_dir}" if has_labels else ""))


def write_dataset_json(dataset_dir: Path, num_training: int, has_test: bool):
    """nnU-Net v2 dataset.json format."""
    labels_for_json = {v: k for k, v in LABEL_MAP.items()}  # name -> id, id 0 required = background
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": labels_for_json,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
        "description": (
            "OAI-ZIB knee MRI segmentation: femur, femoral cartilage, tibia, "
            "medial tibial cartilage, lateral tibial cartilage. Research use "
            "only — part of an automated segmentation + 3D reconstruction "
            "module; not for clinical deployment."
        ),
    }
    out_path = dataset_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset_json, f, indent=2)
    log(f"Wrote {out_path}")
    return out_path


def main():
    log("=" * 70)
    log("STEP 5/6: BUILD nnU-Net RAW DATASET + dataset.json")
    log("=" * 70)

    report = load_verified_report()
    require(report.get("train") is not None,
            "Integrity report has no 'train' entry — re-run 01_dataset_verification.py")

    dataset_dir = NNUNET_RAW_DIR / NNUNET_FULL_NAME
    images_tr_out = dataset_dir / "imagesTr"
    labels_tr_out = dataset_dir / "labelsTr"
    images_ts_out = dataset_dir / "imagesTs"

    ensure_dirs([images_tr_out, labels_tr_out])

    # Resolve source dirs the same way verification did (handles zip-or-folder)
    src_images_tr = verify_mod.resolve_input(IMAGES_TR, DATA_DIR, "imagesTr")
    src_labels_tr = verify_mod.resolve_input(LABELS_TR, DATA_DIR, "labelsTr")

    train_ids = report["train"]["case_ids"]
    copy_split(train_ids, src_images_tr, src_labels_tr, images_tr_out, labels_tr_out, has_labels=True)

    has_test = report.get("test") is not None
    if has_test:
        src_images_ts = verify_mod.resolve_input(IMAGES_TS, DATA_DIR, "imagesTs")
        src_labels_ts = verify_mod.resolve_input(LABELS_TS, DATA_DIR, "labelsTs")
        test_ids = report["test"]["case_ids"]
        # nnU-Net's imagesTs holds images ONLY (labels used separately, by us,
        # for final evaluation after nnUNetv2_predict — never during training).
        copy_split(test_ids, src_images_ts, src_labels_ts, images_ts_out, None, has_labels=False)
    else:
        log("No test set available yet — imagesTs/ will not be created. "
            "Re-run this script after adding IMAGES_TS/LABELS_TS to config.py.",
            level="WARNING")

    write_dataset_json(dataset_dir, num_training=len(train_ids), has_test=has_test)

    log("=" * 70)
    log(f"nnU-Net raw dataset ready at: {dataset_dir}")
    log("Next: run nnU-Net's own integrity check (Step 7), then planning "
        "and preprocessing (Step 8). See 03_run_nnunet_pipeline.py.")
    log("=" * 70)


if __name__ == "__main__":
    main()
