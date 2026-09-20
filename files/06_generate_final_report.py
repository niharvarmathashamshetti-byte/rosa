"""
06_generate_final_report.py
=============================
STEP 23 of the spec (Section 23).

Reads back the JSON/CSV artifacts produced by the earlier scripts and
assembles final_results/final_report.txt. Every number in the report is
read from a saved file that a previous script produced by actually
running something on your machine. If a file is missing, that section
explicitly says "NOT AVAILABLE — <reason>" rather than inventing a
number, per spec Section 23 ("Never invent or estimate a result").
"""

import json
from pathlib import Path
from datetime import datetime

from config import (
    FINAL_RESULTS_DIR, FINAL_METRICS_DIR, INTEGRITY_REPORT_JSON,
    FINAL_REPORT_TXT, FINAL_MESHES_DIR, LABEL_MAP,
)
from utils import log


def load_json_safe(path: Path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_metric_block(summary: dict, indent="    "):
    if summary is None:
        return f"{indent}NOT AVAILABLE\n"
    lines = []
    for name, m in summary.items():
        if name == "mean_foreground_dice":
            continue
        hd95_str = f"{m['mean_hd95']:.3f}" if m.get("mean_hd95") is not None else \
            f"N/A ({m.get('hd95_available_for_n_cases', 0)}/{m.get('n_cases', 0)} cases had a computable HD95)"
        lines.append(
            f"{indent}{name}: Dice={m['mean_dice']:.4f}  IoU={m['mean_iou']:.4f}  "
            f"Precision={m['mean_precision']:.4f}  Recall={m['mean_recall']:.4f}  HD95={hd95_str}"
        )
    if "mean_foreground_dice" in summary:
        lines.append(f"{indent}MEAN FOREGROUND DICE: {summary['mean_foreground_dice']:.4f}")
    return "\n".join(lines) + "\n"


def main():
    integrity = load_json_safe(INTEGRITY_REPORT_JSON)
    env_report = load_json_safe(FINAL_RESULTS_DIR / "environment_report.json")
    val_summary = load_json_safe(FINAL_METRICS_DIR / "validation_metrics_summary.json")
    test_summary = load_json_safe(FINAL_METRICS_DIR / "test_metrics_summary.json")

    lines = []
    lines.append("=" * 72)
    lines.append("KNEE MRI SEGMENTATION — FINAL REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 72)

    lines.append("\n--- SCOPE / LIMITATIONS ---")
    lines.append(
        "This is a research segmentation + 3D-reconstruction module only. "
        "It has NOT been clinically validated, is NOT integrated with the "
        "ROSA robot, and makes no claims of surgical autonomy, clinical "
        "decision-making, patient safety, or superiority over existing "
        "clinical systems."
    )

    lines.append("\n--- DATASET ---")
    if integrity is None:
        lines.append("NOT AVAILABLE — run 01_dataset_verification.py")
    else:
        train = integrity.get("train")
        test = integrity.get("test")
        lines.append(f"Training cases: {train['num_cases'] if train else 'N/A'}")
        lines.append(f"Official test cases: {test['num_cases'] if test else 'NOT PROVIDED YET'}")
        if train:
            lines.append(f"Image shapes observed (train): {train['unique_shapes_seen']}")
            lines.append(f"Voxel spacing (train) — min: {train['spacing_min']}, "
                          f"max: {train['spacing_max']}, mean: {train['spacing_mean']}")
            lines.append(f"Label values observed (train): {train['all_unique_label_values_seen']} "
                         f"(expected: {sorted(LABEL_MAP.keys())})")
            lines.append(f"Cases with empty foreground mask (train): {train['empty_mask_cases']}")
        lines.append("Integrity status: PASSED (pipeline would have stopped otherwise)")

    lines.append("\n--- MODEL / ENVIRONMENT ---")
    if env_report is None:
        lines.append("NOT AVAILABLE — run 00_environment_check.py")
    else:
        lines.append(f"PyTorch: {env_report['gpu_info']['torch_version']}")
        lines.append(f"CUDA available: {env_report['gpu_info']['cuda_available']}")
        lines.append(f"CUDA version: {env_report['gpu_info']['cuda_version']}")
        for g in env_report["gpu_info"]["gpus"]:
            lines.append(f"GPU: {g['name']} ({g['total_vram_gb']} GB VRAM)")
        lines.append(f"nnU-Net version: {env_report.get('nnunet_version')}")
    lines.append("Architecture: 3D nnU-Net v2 (configuration: 3d_fullres unless noted otherwise in experiment_log.csv)")

    lines.append("\n--- VALIDATION (cross-validation on training set only) ---")
    lines.append(fmt_metric_block(val_summary))
    if val_summary is None:
        lines.append("    Reason: run 04_evaluate_and_visualize.py --mode validation after training.")

    lines.append("\n--- FINAL TEST (official held-out test set, evaluated ONCE) ---")
    if test_summary is None:
        lines.append("    NOT AVAILABLE — either the test set has not been provided "
                      "(IMAGES_TS/LABELS_TS not set in config.py) or "
                      "04_evaluate_and_visualize.py --mode test has not been run yet. "
                      "This is expected and is NOT a fabricated placeholder.")
    else:
        lines.append(fmt_metric_block(test_summary))

    lines.append("\n--- 3D RECONSTRUCTION ---")
    if FINAL_MESHES_DIR.exists() and any(FINAL_MESHES_DIR.iterdir()):
        case_dirs = [d for d in FINAL_MESHES_DIR.iterdir() if d.is_dir()]
        lines.append(f"Cases reconstructed: {len(case_dirs)}")
        for cd in case_dirs:
            stls = sorted(p.name for p in cd.glob("*.stl"))
            lines.append(f"  {cd.name}: {stls}")
    else:
        lines.append("NOT AVAILABLE — run 05_reconstruct_3d.py after evaluation.")

    lines.append("\n" + "=" * 72)
    report_text = "\n".join(lines)

    FINAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_REPORT_TXT, "w") as f:
        f.write(report_text)

    print(report_text)
    log(f"\nFinal report written to {FINAL_REPORT_TXT}")


if __name__ == "__main__":
    main()
