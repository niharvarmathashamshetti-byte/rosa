"""
05_reconstruct_3d.py
======================
STEP 18 of the training process (spec Section 18).

Downstream demonstration only — takes already-evaluated prediction masks
(from 04_evaluate_and_visualize.py) and converts each foreground
structure into its own STL surface mesh via marching cubes, preserving
true physical voxel spacing (so meshes are metrically correct, not
distorted).

Usage:
    python 05_reconstruct_3d.py --source validation --case oaizib_003
    python 05_reconstruct_3d.py --source test --all
"""

import argparse
from pathlib import Path

import numpy as np
from skimage import measure

from config import (
    NNUNET_RAW_DIR, NNUNET_RESULTS_DIR, NNUNET_FULL_NAME, NNUNET_TRAINER,
    NNUNET_PLANS, NNUNET_CONFIGURATIONS_TO_TRY, FOLDS, LABEL_MAP,
    FOREGROUND_LABELS, FINAL_MESHES_DIR, FINAL_PREDICTIONS_DIR,
)
from utils import log, fail, ensure_dirs, load_nifti

try:
    import trimesh
    HAVE_TRIMESH = True
except ImportError:
    HAVE_TRIMESH = False


def write_stl_numpy_fallback(verts, faces, out_path):
    """Minimal binary STL writer used only if trimesh isn't installed."""
    normals = np.zeros((faces.shape[0], 3), dtype=np.float32)
    with open(out_path, "wb") as f:
        f.write(b"\x00" * 80)  # header
        f.write(np.uint32(faces.shape[0]).tobytes())
        for i, tri in enumerate(faces):
            f.write(normals[i].astype(np.float32).tobytes())
            for idx in tri:
                f.write(verts[idx].astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())


def mask_to_stl(mask: np.ndarray, spacing, out_path: Path, label_name: str):
    if mask.sum() == 0:
        log(f"  {label_name}: mask is empty for this case — no mesh generated.",
            level="WARNING")
        return None
    try:
        verts, faces, normals, values = measure.marching_cubes(
            mask.astype(np.uint8), level=0.5, spacing=spacing
        )
    except Exception as e:
        log(f"  {label_name}: marching_cubes failed: {e}", level="WARNING")
        return None

    ensure_dirs([out_path.parent])
    if HAVE_TRIMESH:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(str(out_path))
    else:
        write_stl_numpy_fallback(verts, faces, out_path)
    log(f"  {label_name}: mesh saved -> {out_path} "
        f"({len(verts)} vertices, {len(faces)} faces)")
    return str(out_path)


def reconstruct_case(case_id, pred_path, out_dir):
    log(f"Reconstructing case {case_id} from {pred_path}")
    pred_data, affine, header = load_nifti(pred_path)
    spacing = header.get_zooms()[:3]

    generated = {}
    for label_id in FOREGROUND_LABELS:
        name = LABEL_MAP[label_id]
        mask = (pred_data == label_id)
        out_path = out_dir / case_id / f"{name}.stl"
        result = mask_to_stl(mask, spacing, out_path, name)
        generated[name] = result
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["validation", "test"])
    parser.add_argument("--case", default=None, help="Specific case id; omit with --all")
    parser.add_argument("--all", action="store_true", help="Reconstruct all available cases")
    args = parser.parse_args()

    if not args.case and not args.all:
        fail("Provide either --case <id> or --all")

    if not HAVE_TRIMESH:
        log("trimesh not installed — falling back to a minimal built-in "
            "STL writer. Install trimesh for more robust mesh export: "
            "pip install trimesh --break-system-packages", level="WARNING")

    if args.source == "test":
        pred_dir = FINAL_PREDICTIONS_DIR / "test"
    else:
        # validation predictions live inside nnU-Net's own results tree
        pred_dir = None  # resolved per-fold below

    ensure_dirs([FINAL_MESHES_DIR])
    reconstructed = {}

    if args.source == "test":
        if pred_dir is None or not pred_dir.exists():
            fail(f"No test predictions found at {pred_dir}. Run "
                 f"04_evaluate_and_visualize.py --mode test first.")
        candidates = sorted(pred_dir.glob("*.nii.gz"))
        if args.case:
            candidates = [p for p in candidates if p.stem.replace(".nii", "") == args.case]
            if not candidates:
                fail(f"Case {args.case} not found in {pred_dir}")
        for p in candidates:
            cid = p.name.replace(".nii.gz", "")
            reconstructed[cid] = reconstruct_case(cid, p, FINAL_MESHES_DIR)
    else:
        found_any = False
        for fold in FOLDS:
            val_dir = (NNUNET_RESULTS_DIR / NNUNET_FULL_NAME /
                       f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIGURATIONS_TO_TRY[0]}" /
                       f"fold_{fold}" / "validation")
            if not val_dir.exists():
                continue
            candidates = sorted(val_dir.glob("*.nii.gz"))
            if args.case:
                candidates = [p for p in candidates if p.stem.replace(".nii", "") == args.case]
            for p in candidates:
                found_any = True
                cid = p.name.replace(".nii.gz", "")
                if cid in reconstructed:
                    continue  # already done from an earlier fold
                reconstructed[cid] = reconstruct_case(cid, p, FINAL_MESHES_DIR)
            if args.case and found_any:
                break
        if not found_any:
            fail("No validation predictions found to reconstruct from. "
                 "Run training and 04_evaluate_and_visualize.py --mode validation first.")

    log(f"3D reconstruction complete for {len(reconstructed)} case(s). "
        f"Meshes under {FINAL_MESHES_DIR}")


if __name__ == "__main__":
    main()
