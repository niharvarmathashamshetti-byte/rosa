"""
utils.py
========
Small shared helpers: logging, NIfTI I/O checks, and a hard-fail helper
so integrity problems stop the pipeline instead of being swallowed.
"""

import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

try:
    import nibabel as nib
except ImportError:
    nib = None


class PipelineError(RuntimeError):
    """Raised for any condition that must STOP the pipeline (per spec:
    'if a critical integrity check fails, STOP TRAINING')."""
    pass


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def fail(msg: str):
    """Hard stop. Never call this to just warn — this raises."""
    log(msg, level="FATAL")
    raise PipelineError(msg)


def require(condition: bool, msg: str):
    if not condition:
        fail(msg)


def ensure_dirs(dir_list):
    for d in dir_list:
        Path(d).mkdir(parents=True, exist_ok=True)


def load_nifti(path: Path):
    """Load a NIfTI file, returning (data_array, affine, header) or
    raising PipelineError with a clear message on failure. Never
    returns a silently-corrupted object."""
    if nib is None:
        fail("nibabel is not installed. Run: pip install nibabel --break-system-packages")
    path = Path(path)
    if not path.exists():
        fail(f"NIfTI file does not exist: {path}")
    try:
        img = nib.load(str(path))
        data = img.get_fdata()
    except Exception as e:
        fail(f"Failed to read NIfTI file {path}: {e}")
    return data, img.affine, img.header


def case_id_from_image_filename(fname: str, channel_suffix: str, extensions):
    """
    'oaizib_001_0000.nii.gz' -> 'oaizib_001'
    Returns None if the filename does not match the expected pattern
    (caller must treat None as a reportable anomaly, not silently skip it).
    """
    name = fname
    for ext in sorted(extensions, key=len, reverse=True):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    else:
        return None
    if not name.endswith(channel_suffix):
        return None
    return name[: -len(channel_suffix)]


def case_id_from_label_filename(fname: str, extensions):
    """'oaizib_001.nii.gz' -> 'oaizib_001'"""
    name = fname
    for ext in sorted(extensions, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return None


def file_sha1_short(path: Path, n_bytes: int = 1_000_000) -> str:
    """Cheap partial hash for duplicate-file detection (not a full
    integrity hash — just enough to flag suspicious exact duplicates)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def save_json(obj, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def print_and_exit_if_failed(step_name: str, exc: Exception):
    log(f"STEP FAILED: {step_name}", level="FATAL")
    log(str(exc), level="FATAL")
    sys.exit(1)
