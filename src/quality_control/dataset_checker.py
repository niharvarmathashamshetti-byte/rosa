"""
Dataset Validation and Reporting Module.

This module provides tools to scan, validate, and report on medical imaging datasets.
It handles various formats (NPZ, NIfTI, DICOM, etc.) and automatically classifies
data as either CT intensity images or segmentation labels based on content heuristics.

Remember:
- Process one case at a time to avoid out-of-memory errors.
- Never modify raw data files in-place.
- Do NOT assume label meanings (e.g., 1=femur, 2=tibia); always verify with documentation.
"""

import os
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


def classify_volume_content(volume: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes a numpy array and classifies it as likely 'segmentation_labels',
    'intensity_image', or 'ambiguous'.
    
    Why we need it:
    Medical datasets often mix CT scans and segmentation masks. Applying intensity
    operations (like windowing) to a mask is mathematically invalid. This function
    provides an automated guess to prevent accidental misuse.
    
    Parameters:
    -----------
    volume : np.ndarray
        The 3D image array to classify.
        
    Returns:
    --------
    dict
        Classification details and reasoning.
    """
    reasons = []
    
    # 1. dtype heuristics
    dtype_str = str(volume.dtype)
    is_integer = np.issubdtype(volume.dtype, np.integer)
    
    # 2. Value range and uniqueness
    min_val = float(np.min(volume))
    max_val = float(np.max(volume))
    
    # Random subsample to compute unique values efficiently if volume is large
    # Flatten and sample up to 1 million voxels
    flat_vol = volume.reshape(-1)
    if len(flat_vol) > 1000000:
        sample = np.random.choice(flat_vol, size=1000000, replace=False)
    else:
        sample = flat_vol
        
    unique_vals = np.unique(sample)
    num_unique = len(unique_vals)
    
    score = 0  # positive -> intensity, negative -> labels
    
    if dtype_str in ['uint8', 'int8', 'bool']:
        score -= 2
        reasons.append("Small integer/boolean dtype strongly suggests labels.")
    elif dtype_str in ['float32', 'float64', 'int32', 'int16']:
        score += 1
        reasons.append("Float/large integer dtype suggests intensity/CT values.")
        
    if num_unique < 20:
        score -= 3
        reasons.append(f"Very few unique values ({num_unique}) strongly suggests labels.")
    elif num_unique > 100:
        score += 3
        reasons.append(f"Many unique values ({num_unique}) strongly suggests intensity.")
        
    if -1100 <= min_val <= -900 and max_val > 100:
        score += 2
        reasons.append(f"Value range [{min_val}, {max_val}] is typical for CT Hounsfield Units.")
    elif min_val == 0 and max_val < 10:
        score -= 2
        reasons.append(f"Narrow range [{min_val}, {max_val}] starting at 0 is typical for masks.")

    if score > 2:
        classification = "intensity_image"
        confidence = "High" if score > 4 else "Medium"
    elif score < -2:
        classification = "segmentation_labels"
        confidence = "High" if score < -4 else "Medium"
    else:
        classification = "ambiguous"
        confidence = "Low"
        reasons.append("Conflicting or insufficient heuristics to definitively classify.")

    reasons.append("CAVEAT: This is automated classification — verify with dataset documentation. UNKNOWN — NEEDS VERIFICATION.")
    
    return {
        'classification': classification,
        'confidence': confidence,
        'reasoning': reasons,
        'num_unique': num_unique if num_unique < 1000 else ">1000",
        'value_range': (min_val, max_val),
        'dtype': dtype_str
    }


def scan_dataset(root_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Recursively scans the root directory for supported medical image files.
    
    Why we need it:
    To understand the scope and formats present in our raw dataset before processing.
    
    Parameters:
    -----------
    root_path : str | Path
        Directory to scan.
        
    Returns:
    --------
    dict
        Dictionary containing file counts and lists of paths for each extension.
    """
    root = Path(root_path)
    extensions = ['.npz', '.dcm', '.nii', '.nii.gz', '.nrrd', '.mha', '.mhd']
    
    results = {
        'counts': {ext: 0 for ext in extensions},
        'files': {ext: [] for ext in extensions}
    }
    
    if not root.exists():
        print(f"Directory {root} does not exist.")
        return results
        
    for ext in extensions:
        # Use rglob to find all matching files. 
        # Note: nii.gz requires special handling so it's not double-counted as .nii
        if ext == '.nii':
            files = [f for f in root.rglob('*.nii') if not f.name.endswith('.nii.gz')]
        else:
            files = list(root.rglob(f'*{ext}'))
            
        results['counts'][ext] = len(files)
        results['files'][ext] = [str(f) for f in files]
        
    print("--- Dataset Scan Report ---")
    for ext, count in results['counts'].items():
        if count > 0:
            print(f"{ext}: {count} files")
            
    return results


def check_single_case(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads one file, extracts metadata, and checks for basic validity.
    
    Why we need it:
    Identifies corrupted or irregular files (e.g., all zeros, weird shapes)
    early in the pipeline, one case at a time.
    
    Parameters:
    -----------
    file_path : str | Path
        Path to the image file.
        
    Returns:
    --------
    dict
        Metadata and status of the file.
    """
    path = Path(file_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    
    metadata = {
        'file_path': str(path),
        'format': path.suffix,
        'file_size_mb': round(file_size_mb, 2),
        'status': 'OK',
        'notes': [],
        'array_key': None,
        'shape': None,
        'dtype': None,
        'min_val': None,
        'max_val': None,
        'num_unique': None,
        'content_type': 'UNKNOWN',
        'spacing': None,
        'origin': None
    }
    
    try:
        if path.suffix == '.npz':
            with np.load(path) as data:
                # We expect the 'x' key based on project info, but let's check what's there
                keys = list(data.keys())
                if 'x' in keys:
                    key = 'x'
                elif len(keys) == 1:
                    key = keys[0]
                else:
                    key = keys[0]
                    metadata['notes'].append(f"Multiple keys found {keys}, using {key}.")
                
                metadata['array_key'] = key
                arr = data[key]
                
                metadata['shape'] = arr.shape
                metadata['dtype'] = str(arr.dtype)
                
                if arr.size == 0:
                    metadata['status'] = 'ERROR'
                    metadata['notes'].append('Zero-sized volume.')
                    return metadata
                    
                metadata['min_val'] = float(np.min(arr))
                metadata['max_val'] = float(np.max(arr))
                
                if metadata['min_val'] == 0 and metadata['max_val'] == 0:
                    metadata['status'] = 'ERROR'
                    metadata['notes'].append('Volume is all zeros.')
                
                if np.isnan(arr).any() or np.isinf(arr).any():
                    metadata['status'] = 'ERROR'
                    metadata['notes'].append('Contains NaN or Inf values.')
                
                classification = classify_volume_content(arr)
                metadata['content_type'] = classification['classification']
                metadata['num_unique'] = classification['num_unique']
                
        elif path.suffix in ['.nii', '.gz', '.nrrd', '.mha', '.mhd']:
            # Handle standard medical formats with SimpleITK
            if path.suffix == '.gz' and not path.name.endswith('.nii.gz'):
                raise ValueError("Unsupported .gz file (not .nii.gz)")
            
            if path.name.endswith('.nii.gz'):
                metadata['format'] = '.nii.gz'
                
            img = sitk.ReadImage(str(path))
            
            metadata['shape'] = img.GetSize()  # Note: sitk size is (x, y, z)
            metadata['spacing'] = img.GetSpacing()
            metadata['origin'] = img.GetOrigin()
            
            # Convert to numpy for content checking
            arr = sitk.GetArrayFromImage(img) # arr shape will be (z, y, x)
            metadata['dtype'] = str(arr.dtype)
            
            if arr.size == 0:
                metadata['status'] = 'ERROR'
                metadata['notes'].append('Zero-sized volume.')
                return metadata
                
            metadata['min_val'] = float(np.min(arr))
            metadata['max_val'] = float(np.max(arr))
            
            if metadata['min_val'] == 0 and metadata['max_val'] == 0:
                metadata['status'] = 'ERROR'
                metadata['notes'].append('Volume is all zeros.')
                
            classification = classify_volume_content(arr)
            metadata['content_type'] = classification['classification']
            metadata['num_unique'] = classification['num_unique']
            
        else:
            metadata['status'] = 'WARNING'
            metadata['notes'].append(f"Format {path.suffix} not fully parsed yet.")
            
    except Exception as e:
        metadata['status'] = 'ERROR'
        metadata['notes'].append(f"Failed to load: {str(e)}")
        
    return metadata


def generate_dataset_report(root_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Scans all files and generates a comprehensive CSV report.
    
    Why we need it:
    Provides a bird's eye view of the dataset quality and inconsistencies.
    
    Parameters:
    -----------
    root_path : str | Path
        Directory to scan.
    output_path : str | Path | None
        Where to save the CSV.
        
    Returns:
    --------
    pd.DataFrame
        The generated report dataframe.
    """
    root = Path(root_path)
    if output_path is None:
        output_dir = root / 'outputs' / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'dataset_report.csv'
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
    scan_results = scan_dataset(root)
    all_files = []
    for ext_files in scan_results['files'].values():
        all_files.extend(ext_files)
        
    report_rows = []
    
    for f in tqdm(all_files, desc="Checking dataset cases"):
        f_path = Path(f)
        # simplistic case ID based on filename
        case_id = f_path.stem.replace('.nii', '') 
        
        info = check_single_case(f_path)
        info['case_id'] = case_id
        
        # Flatten notes list into a string
        info['notes'] = " | ".join(info['notes'])
        
        report_rows.append(info)
        
    df = pd.DataFrame(report_rows)
    
    # Reorder columns
    cols = ['case_id', 'file_path', 'format', 'array_key', 'shape', 'dtype', 
            'min_val', 'max_val', 'num_unique', 'content_type', 'file_size_mb', 
            'status', 'notes']
    # Add any extra columns that exist
    cols.extend([c for c in df.columns if c not in cols])
    
    if not df.empty:
        df = df[cols]
        df.to_csv(output_path, index=False)
        print(f"Report saved to {output_path}")
        
    return df


def check_image_mask_pair(image_path: Union[str, Path], mask_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Checks if an image and its corresponding segmentation mask align spatially.
    
    Why we need it:
    A mask that doesn't align with its CT image (different shape, origin, or spacing)
    will cause silent errors during model training.
    
    Parameters:
    -----------
    image_path : str | Path
    mask_path : str | Path
    
    Returns:
    --------
    dict
        Match status and alignment details.
    """
    res = {
        'match': False,
        'shape_match': None,
        'spacing_match': None,
        'origin_match': None,
        'details': []
    }
    
    img_info = check_single_case(image_path)
    mask_info = check_single_case(mask_path)
    
    if img_info['status'] == 'ERROR' or mask_info['status'] == 'ERROR':
        res['details'].append("One or both files have an ERROR status.")
        return res
        
    # Check shapes
    res['shape_match'] = (img_info['shape'] == mask_info['shape'])
    if not res['shape_match']:
        res['details'].append(f"Shape mismatch: {img_info['shape']} vs {mask_info['shape']}")
        
    # Check spatial metadata if available (SimpleITK loaded images)
    if img_info.get('spacing') and mask_info.get('spacing'):
        # using np.allclose to account for minor floating point differences
        res['spacing_match'] = np.allclose(img_info['spacing'], mask_info['spacing'], atol=1e-4)
        if not res['spacing_match']:
            res['details'].append(f"Spacing mismatch: {img_info['spacing']} vs {mask_info['spacing']}")
            
    if img_info.get('origin') and mask_info.get('origin'):
        res['origin_match'] = np.allclose(img_info['origin'], mask_info['origin'], atol=1e-4)
        if not res['origin_match']:
            res['details'].append(f"Origin mismatch: {img_info['origin']} vs {mask_info['origin']}")
            
    # Overall match
    res['match'] = res['shape_match']
    if res['spacing_match'] is not None:
        res['match'] = res['match'] and res['spacing_match']
    if res['origin_match'] is not None:
        res['match'] = res['match'] and res['origin_match']
        
    if res['match']:
        res['details'].append("Files match spatially.")
        
    return res


def print_dataset_summary(report_df: pd.DataFrame) -> None:
    """
    Prints a formatted text summary of the dataset from the report dataframe.
    
    Why we need it:
    Provides a quick, readable summary of what we are dealing with.
    """
    if report_df.empty:
        print("Empty report DataFrame.")
        return
        
    print("\n" + "="*50)
    print(" DATASET SUMMARY")
    print("="*50)
    
    num_cases = len(report_df['case_id'].unique())
    print(f"Total Cases (unique IDs): {num_cases}")
    print(f"Total Volumes (files): {len(report_df)}")
    
    print("\nFormats Found:")
    for fmt, count in report_df['format'].value_counts().items():
        print(f"  {fmt}: {count}")
        
    print("\nContent Types Identified:")
    for ctype, count in report_df['content_type'].value_counts().items():
        print(f"  {ctype}: {count}")
        
    # Shape statistics
    # Convert string tuples back to tuples if read from CSV
    shapes = report_df['shape'].dropna().astype(str).tolist()
    if shapes:
        print("\nShape Information:")
        print(f"  Unique shapes found: {len(set(shapes))}")
        print(f"  Most common shape: {pd.Series(shapes).mode().iloc[0]}")
        
    # Warnings and Errors
    errors = report_df[report_df['status'] == 'ERROR']
    warnings = report_df[report_df['status'] == 'WARNING']
    
    if not errors.empty or not warnings.empty:
        print("\n--- ISSUES DETECTED ---")
        if not errors.empty:
            print(f"Errors found in {len(errors)} files!")
            for _, row in errors.iterrows():
                print(f"  [ERROR] {row['case_id']}: {row['notes']}")
        if not warnings.empty:
            print(f"Warnings found in {len(warnings)} files.")
            for _, row in warnings.iterrows():
                print(f"  [WARNING] {row['case_id']}: {row['notes']}")
    else:
        print("\nNo errors or warnings detected. Data looks clean!")
        
    print("="*50 + "\n")
