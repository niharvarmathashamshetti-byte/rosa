"""
04_evaluate_and_visualize.py
==============================
STEPS 12-17 of the training process (spec Section 14, 15, 16, 17).

Two modes:
  --mode validation   -> uses nnU-Net's own cross-validation predictions
                          (nnUNet_preprocessed/<dataset>/.../validation)
                          Safe to run anytime after training. This is
                          what you use to SELECT the final model
                          (spec Section 12/13 — never the test set).
  --mode test          -> runs nnUNetv2_predict on the OFFICIAL held-out
                          imagesTs and evaluates against labelsTs.
                          Spec Section 14 Step 17: this must be run
                          EXACTLY ONCE, after the final model is chosen.
                          This script will refuse to run in test mode
                          more than once unless --force is passed, and
                          logs a loud warning either way.

Computes per-class Dice, IoU, Precision, Recall, and HD95 (HD95 only if
`medpy` is installed and the class is present in both prediction and
ground truth for that case — otherwise explicitly reported as N/A, per
spec Section 15/23: "If a metric cannot be calculated, explicitly state
why.").

Produces:
  final_results/metrics/<mode>_metrics_per_case.csv
  final_results/metrics/<mode>_metrics_summary.json
  final_results/visualizations/<mode>_case_<id>_slice.png  (several cases,
      including both good and poor performers, per spec Section 17)
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR,
    NNUNET_DATASET_ID, NNUNET_FULL_NAME, NNUNET_TRAINER, NNUNET_PLANS,
    NNUNET_CONFIGURATIONS_TO_TRY, FOLDS, LABEL_MAP, FOREGROUND_LABELS,
    FINAL_METRICS_DIR, FINAL_PREDICTIONS_DIR, FINAL_VISUALIZATIONS_DIR,
)
from utils import log, fail, ensure_dirs, load_nifti, save_json

try:
    from medpy.metric.binary import hd95 as medpy_hd95
    HAVE_MEDPY = True
except ImportError:
    HAVE_MEDPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


TEST_RUN_MARKER = FINAL_METRICS_DIR / ".test_evaluation_already_run"


def set_nnunet_env():
    os.environ["nnUNet_raw"] = str(NNUNET_RAW_DIR)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED_DIR)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS_DIR)


def dice_iou_prec_rec(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else np.nan
    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else np.nan
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    return dice, iou, precision, recall


def compute_hd95(pred_mask, gt_mask, spacing):
    if not HAVE_MEDPY:
        return None, "medpy not installed"
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return None, "class absent in prediction or ground truth for this case"
    try:
        val = medpy_hd95(pred_mask, gt_mask, voxelspacing=spacing)
        return float(val), None
    except Exception as e:
        return None, f"medpy error: {e}"


def evaluate_case(case_id, pred_path, gt_path):
    pred_data, pred_affine, pred_header = load_nifti(pred_path)
    gt_data, gt_affine, gt_header = load_nifti(gt_path)

    if pred_data.shape != gt_data.shape:
        fail(f"Case {case_id}: prediction shape {pred_data.shape} != "
             f"ground truth shape {gt_data.shape}. Cannot evaluate.")

    spacing = gt_header.get_zooms()
    rows = []
    for label_id in FOREGROUND_LABELS:
        pred_bin = (pred_data == label_id).astype(np.uint8)
        gt_bin = (gt_data == label_id).astype(np.uint8)
        dice, iou, prec, rec = dice_iou_prec_rec(pred_bin, gt_bin)
        hd95_val, hd95_note = compute_hd95(pred_bin, gt_bin, spacing)
        rows.append({
            "case_id": case_id,
            "class_id": label_id,
            "class_name": LABEL_MAP[label_id],
            "dice": dice,
            "iou": iou,
            "precision": prec,
            "recall": rec,
            "hd95": hd95_val,
            "hd95_note": hd95_note,
        })
    return rows


def make_visualization(case_id, image_path, gt_path, pred_path, out_dir, tag):
    if not HAVE_MPL:
        log("matplotlib not installed — skipping visualization for "
            f"{case_id} (metrics were still computed).", level="WARNING")
        return None

    img_data, _, _ = load_nifti(image_path)
    gt_data, _, _ = load_nifti(gt_path)
    pred_data, _, _ = load_nifti(pred_path)

    # pick the axial slice with the most foreground in the ground truth
    fg_per_slice = (gt_data > 0).sum(axis=(0, 1))
    z = int(np.argmax(fg_per_slice)) if fg_per_slice.max() > 0 else img_data.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_data[:, :, z].T, cmap="gray", origin="lower")
    axes[0].set_title("MRI slice")

    axes[1].imshow(gt_data[:, :, z].T, cmap="tab10", vmin=0, vmax=5, origin="lower")
    axes[1].set_title("Ground truth")

    axes[2].imshow(pred_data[:, :, z].T, cmap="tab10", vmin=0, vmax=5, origin="lower")
    axes[2].set_title("Prediction")

    diff = (gt_data[:, :, z] != pred_data[:, :, z]).astype(np.uint8)
    axes[3].imshow(img_data[:, :, z].T, cmap="gray", origin="lower")
    axes[3].imshow(np.ma.masked_where(diff.T == 0, diff.T), cmap="autumn", alpha=0.7, origin="lower")
    axes[3].set_title("Disagreement (GT vs pred)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{tag}: case {case_id}, slice z={z}")
    ensure_dirs([out_dir])
    out_path = Path(out_dir) / f"{tag}_case_{case_id}_slice.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def summarize(df: pd.DataFrame):
    summary = {}
    for label_id in FOREGROUND_LABELS:
        name = LABEL_MAP[label_id]
        sub = df[df["class_id"] == label_id]
        summary[name] = {
            "mean_dice": float(sub["dice"].mean(skipna=True)),
            "mean_iou": float(sub["iou"].mean(skipna=True)),
            "mean_precision": float(sub["precision"].mean(skipna=True)),
            "mean_recall": float(sub["recall"].mean(skipna=True)),
            "mean_hd95": (float(sub["hd95"].mean(skipna=True))
                          if sub["hd95"].notna().any() else None),
            "hd95_available_for_n_cases": int(sub["hd95"].notna().sum()),
            "n_cases": int(len(sub)),
        }
    summary["mean_foreground_dice"] = float(
        df.groupby("class_id")["dice"].mean().mean()
    )
    return summary


def run_prediction(images_dir, output_dir, folds):
    ensure_dirs([output_dir])
    cmd = [
        "nnUNetv2_predict",
        "-i", str(images_dir),
        "-o", str(output_dir),
        "-d", str(NNUNET_DATASET_ID),
        "-c", NNUNET_CONFIGURATIONS_TO_TRY[0],
        "-tr", NNUNET_TRAINER,
        "-p", NNUNET_PLANS,
        "-f", *[str(f) for f in folds],
    ]
    log(f"RUNNING: {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=os.environ)
    if proc.returncode != 0:
        fail(f"nnUNetv2_predict failed (exit code {proc.returncode}). "
             f"See console output above for the real error.")


def mode_validation(args):
    """
    Uses nnU-Net's built-in cross-validation predictions, which it writes
    to nnUNet_preprocessed/<dataset>/<plans>__<config>/fold_X/validation/
    during nnUNetv2_train. These are predictions on held-out validation
    folds within the TRAINING set only — never the official test set.
    """
    from config import NNUNET_FULL_NAME as full_name
    labels_tr_dir = NNUNET_RAW_DIR / full_name / "labelsTr"
    images_tr_dir = NNUNET_RAW_DIR / full_name / "imagesTr"

    all_rows = []
    viz_paths = []
    for fold in args.folds:
        val_dir = (NNUNET_RESULTS_DIR / full_name /
                   f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIGURATIONS_TO_TRY[0]}" /
                   f"fold_{fold}" / "validation")
        if not val_dir.exists():
            log(f"No validation predictions found for fold {fold} at {val_dir}. "
                f"Did training for this fold complete? Skipping.", level="WARNING")
            continue
        pred_files = sorted(val_dir.glob("*.nii.gz"))
        log(f"Fold {fold}: found {len(pred_files)} validation predictions")
        for pred_path in pred_files:
            case_id = pred_path.name.replace(".nii.gz", "")
            gt_path = labels_tr_dir / f"{case_id}.nii.gz"
            if not gt_path.exists():
                log(f"No ground truth for validation case {case_id} — skipping.", level="WARNING")
                continue
            all_rows.extend(evaluate_case(case_id, pred_path, gt_path))

    if not all_rows:
        fail("No validation cases were evaluated. Check that training "
             "completed for at least one fold and produced validation "
             "predictions.")

    df = pd.DataFrame(all_rows)
    ensure_dirs([FINAL_METRICS_DIR])
    df.to_csv(FINAL_METRICS_DIR / "validation_metrics_per_case.csv", index=False)
    summary = summarize(df)
    save_json(summary, FINAL_METRICS_DIR / "validation_metrics_summary.json")
    log(json.dumps(summary, indent=2))

    # visualize best + worst mean-foreground-dice case
    per_case_mean = df.groupby("case_id")["dice"].mean().sort_values()
    worst_case = per_case_mean.index[0]
    best_case = per_case_mean.index[-1]
    for tag, cid in [("validation_best", best_case), ("validation_worst", worst_case)]:
        img_path = images_tr_dir / f"{cid}_0000.nii.gz"
        gt_path = labels_tr_dir / f"{cid}.nii.gz"
        # find corresponding prediction file across folds
        pred_path = None
        for fold in args.folds:
            candidate = (NNUNET_RESULTS_DIR / NNUNET_FULL_NAME /
                         f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIGURATIONS_TO_TRY[0]}" /
                         f"fold_{fold}" / "validation" / f"{cid}.nii.gz")
            if candidate.exists():
                pred_path = candidate
                break
        if pred_path:
            p = make_visualization(cid, img_path, gt_path, pred_path,
                                    FINAL_VISUALIZATIONS_DIR, tag)
            if p:
                viz_paths.append(p)

    log(f"Validation evaluation complete. Mean foreground Dice: "
        f"{summary['mean_foreground_dice']:.4f}")
    log(f"Visualizations: {viz_paths}")


def mode_test(args):
    """Spec Step 17: run the FINAL model exactly once on the official
    held-out test set."""
    if TEST_RUN_MARKER.exists() and not args.force:
        fail(f"Test-set evaluation has ALREADY been run once "
             f"(marker: {TEST_RUN_MARKER}). Per spec Section 12/14, the "
             f"official test set must be evaluated exactly ONCE with the "
             f"final chosen model. Re-running changes the scientific "
             f"validity of your reported result. Pass --force only if "
             f"you are certain you want to override this (e.g. you are "
             f"redoing this deliberately and will disclose it).")

    full_name = NNUNET_FULL_NAME
    images_ts_dir = NNUNET_RAW_DIR / full_name / "imagesTs"
    if not images_ts_dir.exists():
        fail(f"{images_ts_dir} does not exist. You have not provided "
             f"IMAGES_TS/LABELS_TS in config.py and re-run "
             f"02_convert_to_nnunet_format.py yet. The official test "
             f"evaluation cannot run without it — this is expected, not "
             f"a bug to work around.")

    from config import LABELS_TS as labels_ts_cfg
    labels_ts_dir = None
    import importlib
    verify_mod = importlib.import_module("01_dataset_verification")
    if labels_ts_cfg is not None:
        labels_ts_dir = verify_mod.resolve_input(labels_ts_cfg, images_ts_dir.parent, "labelsTs")

    if labels_ts_dir is None:
        fail("LABELS_TS is not set — cannot compute test metrics without "
             "ground truth. (Predictions could still be generated, but "
             "spec Section 15 requires Dice/IoU/Precision/Recall, which "
             "need ground truth labels.)")

    run_prediction(images_ts_dir, FINAL_PREDICTIONS_DIR / "test", args.folds)

    all_rows = []
    for pred_path in sorted((FINAL_PREDICTIONS_DIR / "test").glob("*.nii.gz")):
        case_id = pred_path.name.replace(".nii.gz", "")
        gt_path = Path(labels_ts_dir) / f"{case_id}.nii.gz"
        if not gt_path.exists():
            log(f"No ground truth for test case {case_id} — skipping.", level="WARNING")
            continue
        all_rows.extend(evaluate_case(case_id, pred_path, gt_path))

    if not all_rows:
        fail("No test cases were evaluated — check predictions and ground truth paths.")

    df = pd.DataFrame(all_rows)
    ensure_dirs([FINAL_METRICS_DIR])
    df.to_csv(FINAL_METRICS_DIR / "test_metrics_per_case.csv", index=False)
    summary = summarize(df)
    save_json(summary, FINAL_METRICS_DIR / "test_metrics_summary.json")
    log(json.dumps(summary, indent=2))

    per_case_mean = df.groupby("case_id")["dice"].mean().sort_values()
    images_ts_glob = {p.name.replace("_0000.nii.gz", ""): p for p in images_ts_dir.glob("*.nii.gz")}
    for tag, cid in [("test_best", per_case_mean.index[-1]), ("test_worst", per_case_mean.index[0])]:
        img_path = images_ts_glob.get(cid)
        gt_path = Path(labels_ts_dir) / f"{cid}.nii.gz"
        pred_path = (FINAL_PREDICTIONS_DIR / "test" / f"{cid}.nii.gz")
        if img_path and pred_path.exists():
            make_visualization(cid, img_path, gt_path, pred_path, FINAL_VISUALIZATIONS_DIR, tag)

    ensure_dirs([FINAL_METRICS_DIR])
    TEST_RUN_MARKER.write_text("Test evaluation run — see test_metrics_summary.json")
    log(f"TEST evaluation complete and marked as run "
        f"(mean foreground Dice: {summary['mean_foreground_dice']:.4f}). "
        f"Marker file: {TEST_RUN_MARKER}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["validation", "test"])
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS)
    parser.add_argument("--force", action="store_true",
                         help="Override the one-time-only guard on --mode test")
    args = parser.parse_args()

    set_nnunet_env()
    ensure_dirs([FINAL_METRICS_DIR, FINAL_VISUALIZATIONS_DIR, FINAL_PREDICTIONS_DIR])

    if not HAVE_MEDPY:
        log("medpy is not installed — HD95 will be reported as N/A for "
            "all classes/cases. Install with: "
            "pip install medpy --break-system-packages", level="WARNING")

    if args.mode == "validation":
        mode_validation(args)
    else:
        mode_test(args)


if __name__ == "__main__":
    main()
