"""
config.py
=========
Single source of truth for paths, label mapping, and nnU-Net dataset
identity. Edit ONLY this file to point at your data; every other script
imports from here so there is exactly one place that can be wrong.

All paths are read as given (folder OR .zip) — 01_dataset_verification.py
will detect which one you gave it and extract zips automatically without
touching your originals.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# 1. RAW INPUT PATHS  (folders or .zip files — either is accepted)
# ---------------------------------------------------------------------
# Update these if you move the data. Currently set from what you provided.
IMAGES_TR = Path(r"C:\Users\91950\Downloads\imagesTr")
LABELS_TR = Path(r"C:\Users\91950\Downloads\labelsTr")

# Not yet provided. Leave as None until you have them; the pipeline will
# detect the None and skip the final held-out test stage explicitly
# (it will NOT silently proceed or fabricate a result).
IMAGES_TS = None   # e.g. Path(r"C:\Users\91950\Downloads\imagesTs")
LABELS_TS = None   # e.g. Path(r"C:\Users\91950\Downloads\labelsTs")

# ---------------------------------------------------------------------
# 2. WORKING DIRECTORY LAYOUT
# ---------------------------------------------------------------------
# Everything the pipeline creates lives under PROJECT_ROOT. Your original
# data directories/zips above are never modified or moved.
PROJECT_ROOT = Path(r"C:\Users\91950\Documents\knee_seg_project")

DATA_DIR              = PROJECT_ROOT / "data"                # extracted copies
NNUNET_RAW_DIR         = PROJECT_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED_DIR = PROJECT_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS_DIR     = PROJECT_ROOT / "nnUNet_results"
FINAL_RESULTS_DIR      = PROJECT_ROOT / "final_results"

FINAL_METRICS_DIR       = FINAL_RESULTS_DIR / "metrics"
FINAL_PREDICTIONS_DIR   = FINAL_RESULTS_DIR / "predictions"
FINAL_VISUALIZATIONS_DIR = FINAL_RESULTS_DIR / "visualizations"
FINAL_MESHES_DIR         = FINAL_RESULTS_DIR / "3d_meshes"
FINAL_LOGS_DIR           = FINAL_RESULTS_DIR / "training_logs"
EXPERIMENT_LOG_CSV       = FINAL_RESULTS_DIR / "experiment_log.csv"
FINAL_REPORT_TXT         = FINAL_RESULTS_DIR / "final_report.txt"
INTEGRITY_REPORT_JSON    = FINAL_RESULTS_DIR / "dataset_integrity_report.json"

ALL_DIRS = [
    DATA_DIR, NNUNET_RAW_DIR, NNUNET_PREPROCESSED_DIR, NNUNET_RESULTS_DIR,
    FINAL_RESULTS_DIR, FINAL_METRICS_DIR, FINAL_PREDICTIONS_DIR,
    FINAL_VISUALIZATIONS_DIR, FINAL_MESHES_DIR, FINAL_LOGS_DIR,
]

# ---------------------------------------------------------------------
# 3. nnU-Net DATASET IDENTITY
# ---------------------------------------------------------------------
# nnU-Net v2 requires a numeric dataset ID (1-999) and a name.
# Folder created will be: DatasetXXX_KneeOAIZIB
NNUNET_DATASET_ID = 501
NNUNET_DATASET_NAME = "KneeOAIZIB"
NNUNET_FULL_NAME = f"Dataset{NNUNET_DATASET_ID:03d}_{NNUNET_DATASET_NAME}"

# ---------------------------------------------------------------------
# 4. LABEL MAP — DO NOT EDIT without re-verifying against your data.
#    This must match Section 6 of the project spec exactly.
# ---------------------------------------------------------------------
LABEL_MAP = {
    0: "background",
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}
FOREGROUND_LABELS = [k for k in LABEL_MAP.keys() if k != 0]
EXPECTED_LABEL_SET = set(LABEL_MAP.keys())

# ---------------------------------------------------------------------
# 5. FILE NAMING CONVENTION
# ---------------------------------------------------------------------
# image:  <case>_0000.nii.gz   label: <case>.nii.gz
IMAGE_CHANNEL_SUFFIX = "_0000"
NIFTI_EXTENSIONS = (".nii.gz", ".nii")

# ---------------------------------------------------------------------
# 6. TRAINING / EXPERIMENT SETTINGS
# ---------------------------------------------------------------------
RANDOM_SEED = 42
NNUNET_CONFIGURATIONS_TO_TRY = ["3d_fullres"]  # extend later (e.g. 3d_lowres) if GPU-limited
NNUNET_TRAINER = "nnUNetTrainer"                # default trainer; do not silently swap
NNUNET_PLANS = "nnUNetPlans"
FOLDS = [0, 1, 2, 3, 4]                          # 5-fold CV, nnU-Net default
