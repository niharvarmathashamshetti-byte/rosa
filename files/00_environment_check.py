"""
00_environment_check.py
========================
STEP 1 of the training process (spec Section 14 / 20 / 21).

Detects: Python version, PyTorch version, CUDA availability, GPU model,
VRAM, nnU-Net installation. Prints a report and records it to
final_results/ for reproducibility. Does NOT guess your hardware —
everything here is read live from your machine when you run this.

Run this FIRST, before anything else, on your local machine:
    python 00_environment_check.py
"""

import sys
import platform
import subprocess
from pathlib import Path

from config import FINAL_RESULTS_DIR
from utils import log, save_json, ensure_dirs


def get_gpu_info():
    info = {"cuda_available": False, "gpus": [], "cuda_version": None,
            "torch_version": None}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": props.multi_processor_count,
                })
    except ImportError:
        log("PyTorch is not installed. Install it before proceeding "
            "(see https://pytorch.org for the correct CUDA build for "
            "your system).", level="ERROR")
    return info


def check_nnunet_installed():
    try:
        import nnunetv2  # noqa: F401
        version = getattr(nnunetv2, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_env_vars():
    import os
    keys = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
    return {k: os.environ.get(k) for k in keys}


def main():
    ensure_dirs([FINAL_RESULTS_DIR])
    log("=" * 70)
    log("STEP 1: ENVIRONMENT VERIFICATION")
    log("=" * 70)

    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    gpu_info = get_gpu_info()
    report["gpu_info"] = gpu_info

    nnunet_ok, nnunet_version = check_nnunet_installed()
    report["nnunet_installed"] = nnunet_ok
    report["nnunet_version"] = nnunet_version

    report["nnunet_env_vars"] = check_env_vars()

    log(f"Python: {report['python_version'].splitlines()[0]}")
    log(f"Platform: {report['platform']}")
    log(f"PyTorch: {gpu_info['torch_version']}")
    log(f"CUDA available: {gpu_info['cuda_available']}")
    log(f"CUDA version: {gpu_info['cuda_version']}")
    for g in gpu_info["gpus"]:
        log(f"  GPU {g['index']}: {g['name']} — {g['total_vram_gb']} GB VRAM")

    if not gpu_info["cuda_available"]:
        log("No CUDA-capable GPU detected. 3D nnU-Net training on CPU is "
            "not practically feasible for this dataset size. Fix your "
            "PyTorch/CUDA install before proceeding.", level="WARNING")

    if not nnunet_ok:
        log("nnunetv2 is NOT installed. Install with:\n"
            "    pip install nnunetv2 --break-system-packages\n"
            "or in a virtualenv without that flag.", level="ERROR")
    else:
        log(f"nnunetv2 version: {nnunet_version}")

    env_vars = report["nnunet_env_vars"]
    missing_env = [k for k, v in env_vars.items() if not v]
    if missing_env:
        log(f"nnU-Net environment variables not set: {missing_env}. "
            "02_convert_to_nnunet_format.py sets these for the current "
            "process, but for CLI commands (nnUNetv2_plan_and_preprocess, "
            "nnUNetv2_train) run in a NEW terminal, you must set them "
            "yourself — see that script's printed instructions.",
            level="WARNING")

    # Advisory VRAM check — does not change anything automatically,
    # per spec Section 20 ("do not silently change the scientific
    # experiment").
    if gpu_info["gpus"]:
        min_vram = min(g["total_vram_gb"] for g in gpu_info["gpus"])
        if min_vram < 10:
            log(f"Smallest detected GPU has {min_vram} GB VRAM. Standard "
                "3d_fullres nnU-Net configs typically want >=10-11GB. "
                "If training later fails with CUDA OOM, that is an "
                "expected, explained failure mode, not a bug — you will "
                "need to either reduce nnU-Net's batch size / patch size "
                "via a custom plans file, or use the 3d_lowres "
                "configuration. This script will NOT change your "
                "configuration automatically.", level="WARNING")

    save_json(report, FINAL_RESULTS_DIR / "environment_report.json")
    log(f"Environment report saved to "
        f"{FINAL_RESULTS_DIR / 'environment_report.json'}")

    if not (gpu_info["cuda_available"] and nnunet_ok):
        log("Environment check found blocking issues (see ERROR lines "
            "above). Resolve them before running 01_dataset_verification.py.",
            level="FATAL")
        sys.exit(1)

    log("Environment check PASSED.")


if __name__ == "__main__":
    main()
