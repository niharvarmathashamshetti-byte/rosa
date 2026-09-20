"""
03_run_nnunet_pipeline.py
===========================
STEPS 7-11 of the training process (spec Section 14).

This orchestrates nnU-Net v2's OWN CLI tools via subprocess (this is the
correct and supported way to drive nnU-Net — it is not reimplemented
here, per spec Section 11: "Use nnU-Net's planning and preprocessing
mechanisms wherever possible").

Sets the required nnU-Net environment variables for THIS process and
every subprocess it spawns, so you do not need to set them manually in
a fresh terminal (though you still can, if you prefer to run individual
nnUNetv2_* commands by hand later).

Usage:
    python 03_run_nnunet_pipeline.py --step verify
    python 03_run_nnunet_pipeline.py --step preprocess
    python 03_run_nnunet_pipeline.py --step smoketest
    python 03_run_nnunet_pipeline.py --step train --fold 0
    python 03_run_nnunet_pipeline.py --step train_all_folds
    python 03_run_nnunet_pipeline.py --step all --fold 0
"""

import os
import sys
import argparse
import subprocess

from config import (
    NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR,
    NNUNET_DATASET_ID, NNUNET_FULL_NAME, NNUNET_TRAINER,
    NNUNET_CONFIGURATIONS_TO_TRY, FOLDS, FINAL_LOGS_DIR,
)
from utils import log, fail, ensure_dirs


def set_nnunet_env():
    os.environ["nnUNet_raw"] = str(NNUNET_RAW_DIR)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED_DIR)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS_DIR)
    log(f"nnUNet_raw          = {NNUNET_RAW_DIR}")
    log(f"nnUNet_preprocessed = {NNUNET_PREPROCESSED_DIR}")
    log(f"nnUNet_results      = {NNUNET_RESULTS_DIR}")
    log("NOTE: if you run nnUNetv2_* commands manually in a separate "
        "terminal, set these three environment variables yourself first.")


def run(cmd, log_file=None):
    log(f"RUNNING: {' '.join(cmd)}")
    if log_file:
        ensure_dirs([FINAL_LOGS_DIR])
        with open(log_file, "a") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=os.environ)
    else:
        proc = subprocess.run(cmd, env=os.environ)
    if proc.returncode != 0:
        fail(f"Command failed (exit code {proc.returncode}): {' '.join(cmd)}. "
             f"See log above / {log_file if log_file else 'stdout'} for the "
             f"real nnU-Net error message — this pipeline does not mask "
             f"underlying failures.")
    return proc


def step_verify():
    """Step 7: nnU-Net's own dataset fingerprint / integrity check."""
    log("STEP 7: nnU-Net dataset integrity verification")
    run([
        "nnUNetv2_plan_and_preprocess",
        "-d", str(NNUNET_DATASET_ID),
        "--verify_dataset_integrity",
        "-c", *NNUNET_CONFIGURATIONS_TO_TRY,
        "--clean",
    ], log_file=str(FINAL_LOGS_DIR / "step7_verify_and_preprocess.log"))
    log("nnU-Net integrity verification + planning + preprocessing complete "
        "(this single nnUNetv2_plan_and_preprocess call performs Steps 7 and 8).")


def step_smoketest(fold=0):
    """
    Step 9: short smoke test. There is no official nnU-Net 'smoke test'
    flag, so we run a genuinely short training (a handful of iterations)
    by capping epochs via the trainer's env override, purely to confirm
    the pipeline runs end-to-end before committing to full training.
    """
    log("STEP 9: SMOKE TEST — short run to confirm the pipeline executes "
        "end-to-end (NOT a scientifically valid trained model).")
    os.environ["nnUNet_n_proc_DA"] = os.environ.get("nnUNet_n_proc_DA", "4")
    # nnUNetTrainer subclasses exist for short debugging runs in some
    # nnU-Net versions (e.g. nnUNetTrainer_5epochs). If not present in
    # your installed version, this step will fail clearly rather than
    # silently falling back to a full run.
    smoke_trainer = "nnUNetTrainer_5epochs"
    log(f"Attempting smoke test with trainer '{smoke_trainer}' "
        f"(5 epochs). If your nnunetv2 version does not ship this "
        f"trainer class, this step will fail with an explicit error — "
        f"in that case, skip --step smoketest and go straight to "
        f"--step train, or add a custom debug trainer.")
    try:
        run([
            "nnUNetv2_train",
            str(NNUNET_DATASET_ID),
            NNUNET_CONFIGURATIONS_TO_TRY[0],
            str(fold),
            "-tr", smoke_trainer,
        ], log_file=str(FINAL_LOGS_DIR / "step9_smoketest.log"))
        log("Smoke test completed without crashing. Proceeding to full "
            "training is now lower-risk.")
    except SystemExit:
        log("Smoke test trainer unavailable or failed. This does not "
            "block full training — re-run with --step train when ready, "
            "but investigate the log above first if the failure looks "
            "like a real configuration/data problem rather than a "
            "missing-trainer-class issue.", level="WARNING")


def step_train(fold):
    """Step 10 + 11: actual training + checkpoint saving (checkpointing
    is handled internally by nnU-Net into nnUNet_results/)."""
    log(f"STEP 10/11: TRAINING fold {fold} "
        f"(configuration={NNUNET_CONFIGURATIONS_TO_TRY[0]}, trainer={NNUNET_TRAINER})")
    run([
        "nnUNetv2_train",
        str(NNUNET_DATASET_ID),
        NNUNET_CONFIGURATIONS_TO_TRY[0],
        str(fold),
        "-tr", NNUNET_TRAINER,
    ], log_file=str(FINAL_LOGS_DIR / f"step10_train_fold{fold}.log"))
    log(f"Fold {fold} training complete. Checkpoints saved under "
        f"{NNUNET_RESULTS_DIR} (nnU-Net's own checkpointing).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True,
                         choices=["verify", "smoketest", "train", "train_all_folds", "all"])
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    set_nnunet_env()
    ensure_dirs([FINAL_LOGS_DIR])

    if args.step == "verify":
        step_verify()
    elif args.step == "smoketest":
        step_smoketest(args.fold)
    elif args.step == "train":
        step_train(args.fold)
    elif args.step == "train_all_folds":
        for f in FOLDS:
            step_train(f)
    elif args.step == "all":
        step_verify()
        step_smoketest(args.fold)
        step_train(args.fold)


if __name__ == "__main__":
    main()
