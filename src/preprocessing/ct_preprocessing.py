"""
CT Image Preprocessing Module.

NOTE: SKELETON ONLY.
These functions are to be used AFTER thorough data verification.

This module provides standard medical image preprocessing techniques such as 
windowing, resampling, cropping, and normalization.

Remember:
- Do NOT train any models here.
- Never modify raw data. 
- Process one case at a time.
- Verify if the volume is CT intensity or a segmentation label BEFORE applying!
"""

import numpy as np
import SimpleITK as sitk
from typing import Union, Tuple


def apply_window(volume: np.ndarray, window_center: int = 400, window_width: int = 1500) -> np.ndarray:
    """
    Applies CT windowing to an intensity volume.
    
    Why we need it:
    CT scanners measure radiodensity in Hounsfield Units (HU), which can range from 
    -1024 (air) to +3000 (dense bone). Displays and Neural Networks prefer values mapped 
    to a standard range (like [0, 1]). Windowing restricts the HU range to tissues of interest
    (e.g., bone window vs soft tissue window).
    
    WARNING: This ONLY makes sense for CT intensity data. Do NOT apply to segmentation labels!
    
    Parameters:
    -----------
    volume : np.ndarray
        The input CT volume in Hounsfield Units.
    window_center : int
        The midpoint (Level) of the displayed HU range.
    window_width : int
        The total range of HU values to display.
        
    Returns:
    --------
    np.ndarray
        The windowed volume, clipped and scaled to [0, 1].
    """
    print("Applying CT windowing. To be used AFTER data verification.")
    
    # Calculate bounds
    min_val = window_center - (window_width / 2.0)
    max_val = window_center + (window_width / 2.0)
    
    # Clip values outside the window
    windowed = np.clip(volume, min_val, max_val)
    
    # Scale to [0, 1]
    windowed = (windowed - min_val) / window_width
    
    return windowed


def resample_image(image: sitk.Image, target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5), 
                   interpolator: int = sitk.sitkLinear) -> sitk.Image:
    """
    Resamples a SimpleITK Image to a new target physical spacing.
    
    Why we need it:
    Medical scans from different patients/scanners often have different physical resolutions 
    (e.g., 0.5mm vs 1.0mm per voxel). Resampling standardizes this so 1 voxel always represents
    the same physical volume, which is critical for consistent ML model performance.
    
    Why not numpy resize?
    Simple matrix resizing destroys spatial metadata (origin, direction, spacing) which is 
    crucial for relating predictions back to patient anatomy in the real world.
    
    WARNING: Use sitk.sitkLinear for intensity images, but MUST use sitk.sitkNearestNeighbor 
    for segmentation masks to avoid creating false interpolated classes (e.g., class 1.5).
    
    Parameters:
    -----------
    image : sitk.Image
        The input image to resample.
    target_spacing : tuple
        The desired voxel spacing (x, y, z) in millimeters.
    interpolator : int
        SimpleITK interpolator enum.
        
    Returns:
    --------
    sitk.Image
        The resampled image.
    """
    print("Resampling image. To be used AFTER data verification.")
    
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size based on ratio of old/new spacing
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)


def crop_volume(image: sitk.Image, start_index: Tuple[int, int, int], crop_size: Tuple[int, int, int]) -> sitk.Image:
    """
    Extracts a region of interest (ROI) from a medical image.
    
    Why we need it:
    Knee MRI/CT scans often include huge amounts of background (air) or other irrelevant 
    anatomy. Cropping to the knee joint reduces memory usage and focuses the model.
    Using SimpleITK preserves the correct origin, which changes when you crop!
    
    Parameters:
    -----------
    image : sitk.Image
        The input image.
    start_index : tuple
        The starting voxel index (x, y, z) of the crop.
    crop_size : tuple
        The size of the crop (x_size, y_size, z_size).
        
    Returns:
    --------
    sitk.Image
        The cropped image.
    """
    print("Cropping volume. To be used AFTER data verification.")
    
    # Validate bounds
    image_size = image.GetSize()
    for i in range(3):
        if start_index[i] < 0 or start_index[i] >= image_size[i]:
            raise ValueError(f"Start index {start_index} is out of bounds for image size {image_size}.")
        if start_index[i] + crop_size[i] > image_size[i]:
            raise ValueError(f"Crop region extends beyond image bounds along axis {i}.")
            
    cropper = sitk.RegionOfInterestImageFilter()
    cropper.SetIndex(start_index)
    cropper.SetSize(crop_size)
    
    return cropper.Execute(image)


def normalize_intensity(volume: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalizes the intensity values of a volume.
    
    Why we need it:
    Neural networks train better when inputs have zero mean and unit variance, or are 
    scaled to [0, 1].
    
    WARNING: ONLY for intensity data, NOT for segmentation labels!
    
    Parameters:
    -----------
    volume : np.ndarray
        The input intensity volume.
    method : str
        'minmax' for scaling to [0,1], or 'zscore' for mean=0, std=1.
        
    Returns:
    --------
    np.ndarray
        The normalized volume.
    """
    print("Normalizing intensity. To be used AFTER data verification.")
    
    if method == 'minmax':
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val - min_val == 0:
            return np.zeros_like(volume, dtype=np.float32)
        return (volume - min_val) / (max_val - min_val)
        
    elif method == 'zscore':
        mean = np.mean(volume)
        std = np.std(volume)
        if std == 0:
            return np.zeros_like(volume, dtype=np.float32)
        return (volume - mean) / std
        
    else:
        print("UNKNOWN — NEEDS VERIFICATION. Unsupported normalization method.")
        return volume
