import SimpleITK as sitk
from pathlib import Path
from typing import Union, Dict, Any, Tuple

def load_nifti(path: Union[str, Path]) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Loads a NIfTI (.nii or .nii.gz) file using SimpleITK.
    
    Why we need this: NIfTI is a standard medical imaging format that combines 
    image data and physical metadata into a single file (unlike DICOM series).
    
    Args:
        path (Union[str, Path]): Path to the NIfTI file.
        
    Returns:
        Tuple[sitk.Image, Dict[str, Any]]: The loaded image and its metadata.
    """
    path = Path(path)
    print(f"Loading NIfTI file from: {path}")
    
    try:
        # Read the image
        image = sitk.ReadImage(str(path))
        
        # Extract metadata (physical properties of the image)
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()
        
        # Calculate physical dimensions
        physical_size = tuple(sz * sp for sz, sp in zip(size, spacing))
        
        metadata = {
            'size': size,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'pixel_type': pixel_type,
            'physical_size': physical_size
        }
        
        print(f"--- NIfTI Metadata ---")
        for k, v in metadata.items():
            print(f"{k}: {v}")
            
        return image, metadata
        
    except Exception as e:
        print(f"Failed to load NIfTI file {path}. Error: {str(e)}")
        raise

def load_medical_image(path: Union[str, Path]) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Auto-detects the medical image format from the extension and loads it.
    Supports .nii, .nii.gz, .nrrd, .mha, .mhd.
    
    Why we need this: A unified interface allows the rest of our processing 
    pipeline to be agnostic to the specific file format on disk.
    
    Args:
        path (Union[str, Path]): Path to the medical image file.
        
    Returns:
        Tuple[sitk.Image, Dict[str, Any]]: The loaded image and its metadata.
    """
    path = Path(path)
    print(f"Auto-loading medical image from: {path}")
    
    # SimpleITK natively supports these extensions
    supported_extensions = {'.nii', '.nii.gz', '.nrrd', '.mha', '.mhd'}
    
    # Handle .nii.gz as a special case for suffixes
    ext = ''.join(path.suffixes).lower()
    if ext not in supported_extensions and path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file format '{ext}' for file {path}. "
            f"Supported formats are: {supported_extensions}"
        )
        
    try:
        # SimpleITK's ReadImage automatically detects the correct internal reader based on file content/extension
        image = sitk.ReadImage(str(path))
        
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()
        physical_size = tuple(sz * sp for sz, sp in zip(size, spacing))
        
        metadata = {
            'size': size,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'pixel_type': pixel_type,
            'physical_size': physical_size
        }
        
        print(f"--- Medical Image Metadata ---")
        for k, v in metadata.items():
            print(f"{k}: {v}")
            
        return image, metadata
        
    except Exception as e:
        print(f"Failed to load medical image {path}. Error: {str(e)}")
        raise
