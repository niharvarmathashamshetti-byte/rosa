import SimpleITK as sitk
import pydicom
from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
import os

def load_dicom_series(folder: Union[str, Path]) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Finds and loads a DICOM series from a folder into a 3D SimpleITK image.
    
    Why we need this: DICOMs are stored as multiple 2D slice files in a directory. 
    SimpleITK can automatically read these slices, order them by spatial position, 
    and reconstruct the 3D volume.
    
    Args:
        folder (Union[str, Path]): Directory containing DICOM files.
        
    Returns:
        Tuple[sitk.Image, Dict[str, Any]]: The loaded 3D image and its physical metadata.
    """
    folder = Path(folder)
    print(f"Loading DICOM series from: {folder}")
    
    try:
        reader = sitk.ImageSeriesReader()
        
        # Get list of DICOM files in the directory
        dicom_names = reader.GetGDCMSeriesFileNames(str(folder))
        
        if not dicom_names:
            raise ValueError(f"No DICOM series found in directory: {folder}")
            
        reader.SetFileNames(dicom_names)
        
        # Read the 3D volume
        image = reader.Execute()
        
        # Extract physical metadata
        size = image.GetSize()
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        pixel_type = image.GetPixelIDTypeAsString()
        
        # Calculate physical dimensions (in mm)
        physical_size = tuple(sz * sp for sz, sp in zip(size, spacing))
        
        metadata = {
            'size': size,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'pixel_type': pixel_type,
            'physical_size': physical_size
        }
        
        print(f"--- DICOM Series Loaded ---")
        for k, v in metadata.items():
            print(f"{k}: {v}")
            
        return image, metadata
        
    except Exception as e:
        print(f"Failed to load DICOM series from {folder}. Error: {str(e)}")
        raise

def get_dicom_metadata(folder: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads metadata tags from the first DICOM file in a directory using pydicom.
    
    Why we need this: SimpleITK provides physical geometry metadata, but pydicom 
    lets us read clinical tags (patient ID, modality, manufacturer) which are 
    crucial for cohort selection and device analysis.
    
    Args:
        folder (Union[str, Path]): Directory containing DICOM files.
        
    Returns:
        Dict[str, Any]: Dictionary containing selected DICOM tags.
    """
    folder = Path(folder)
    metadata = {}
    
    try:
        # Find first DICOM file (heuristic: ends with .dcm or has no extension)
        first_dcm_path = None
        for file in folder.iterdir():
            if file.is_file() and (file.suffix.lower() == '.dcm' or not file.suffix):
                # Try reading to confirm it's DICOM
                try:
                    pydicom.dcmread(str(file), stop_before_pixels=True)
                    first_dcm_path = file
                    break
                except pydicom.errors.InvalidDicomError:
                    continue
                    
        if not first_dcm_path:
            raise ValueError(f"Could not identify a valid DICOM file in {folder}")
            
        print(f"Reading DICOM tags from: {first_dcm_path}")
        ds = pydicom.dcmread(str(first_dcm_path), stop_before_pixels=True)
        
        # Helper to safely extract tags
        def get_tag(ds: pydicom.dataset.FileDataset, tag_name: str) -> str:
            val = getattr(ds, tag_name, None)
            return str(val) if val is not None else "UNKNOWN — NEEDS VERIFICATION"
            
        metadata = {
            'PatientID': get_tag(ds, 'PatientID'),
            'Modality': get_tag(ds, 'Modality'),
            'StudyDescription': get_tag(ds, 'StudyDescription'),
            'SeriesDescription': get_tag(ds, 'SeriesDescription'),
            'Manufacturer': get_tag(ds, 'Manufacturer'),
            'SliceThickness': get_tag(ds, 'SliceThickness'),
            'PixelSpacing': get_tag(ds, 'PixelSpacing'),
            'Rows': get_tag(ds, 'Rows'),
            'Columns': get_tag(ds, 'Columns')
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error reading DICOM metadata from {folder}: {str(e)}")
        # Return dict with unknown values on failure
        keys = ['PatientID', 'Modality', 'StudyDescription', 'SeriesDescription', 
                'Manufacturer', 'SliceThickness', 'PixelSpacing', 'Rows', 'Columns']
        return {k: "UNKNOWN — NEEDS VERIFICATION" for k in keys}

def find_dicom_series_in_directory(root: Union[str, Path]) -> List[Path]:
    """
    Recursively scans a root directory for folders containing DICOM files.
    
    Why we need this: DICOM datasets are often nested deeply in folder structures 
    (e.g., Patient -> Study -> Series). We need to find the specific folders 
    that actually contain the image files.
    
    Args:
        root (Union[str, Path]): Root directory to search.
        
    Returns:
        List[Path]: List of directories that contain at least one .dcm file.
    """
    root = Path(root)
    dicom_dirs = set()
    
    print(f"Scanning for DICOM directories in: {root}")
    
    try:
        # Search for .dcm files and collect their parent directories
        for file in root.rglob("*.dcm"):
            dicom_dirs.add(file.parent)
            
        result = list(dicom_dirs)
        print(f"Found {len(result)} DICOM series directories.")
        return result
        
    except Exception as e:
        print(f"Error scanning for DICOMs: {str(e)}")
        return []
