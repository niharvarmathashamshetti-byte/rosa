# =============================================================================
# src.io — Medical Image I/O
# =============================================================================
# Loaders for various medical image formats:
#   - npz_loader.py    : NumPy .npz files (our current dataset format)
#   - dicom_loader.py  : DICOM series (.dcm files)
#   - nifti_loader.py  : NIfTI (.nii, .nii.gz), NRRD, MHA formats
# =============================================================================

from .npz_loader import load_npz, inspect_npz, scan_npz_dataset
from .dicom_loader import load_dicom_series, get_dicom_metadata
from .nifti_loader import load_nifti, load_medical_image
