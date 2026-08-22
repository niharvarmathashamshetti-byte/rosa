import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any, List
from tqdm import tqdm

def load_npz(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a single .npz file and extracts information about its contents.
    
    Why we need this: .npz files contain multiple arrays. We need a reliable way 
    to extract the data and understand its properties (dtype, shape) without 
    blindly assuming what's inside.
    
    Args:
        path (Union[str, Path]): Path to the .npz file.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'volume': The numpy array (from key 'x' if present, else the first key)
            - 'keys': List of keys in the npz file
            - 'shape': Shape of the array
            - 'dtype': Data type of the array
            - 'min_val': Minimum value in the array
            - 'max_val': Maximum value in the array
            - 'unique_values': List of unique values if < 50, else None
            - 'num_unique': Number of unique values
    """
    path = Path(path)
    print(f"Loading NPZ file from: {path}")
    
    try:
        # Load the file in read mode. allow_pickle=False for security and standard numpy arrays
        with np.load(path, allow_pickle=False) as data:
            keys = list(data.keys())
            
            # Use 'x' if available, otherwise fallback to the first key
            array_key = 'x' if 'x' in keys else keys[0]
            
            # Load the actual volume into memory
            volume = data[array_key]
            
            shape = volume.shape
            dtype = volume.dtype
            min_val = float(np.min(volume))
            max_val = float(np.max(volume))
            
            # Find unique values. np.unique can be slow for large arrays, so we use it carefully
            unique_vals = np.unique(volume)
            num_unique = len(unique_vals)
            
            if num_unique < 50:
                unique_values_list = unique_vals.tolist()
            else:
                unique_values_list = None
                
            report = {
                'volume': volume,
                'keys': keys,
                'shape': shape,
                'dtype': str(dtype),
                'min_val': min_val,
                'max_val': max_val,
                'unique_values': unique_values_list,
                'num_unique': num_unique
            }
            
            print(f"--- NPZ Inspection Report ---")
            print(f"Keys found: {keys}")
            print(f"Loaded array from key: {array_key}")
            print(f"Shape: {shape}")
            print(f"Dtype: {dtype}")
            print(f"Value range: [{min_val}, {max_val}]")
            print(f"Number of unique values: {num_unique}")
            if unique_values_list is not None:
                print(f"Unique values: {unique_values_list}")
                
            return report
            
    except Exception as e:
        print(f"Failed to load npz file {path}. Error: {str(e)}")
        raise

def inspect_npz(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspects a .npz file without loading the entire volume into memory when possible.
    Heuristically determines whether arrays contain intensity images or segmentation labels.
    
    Why we need this: Loading many large volumes just to check their contents uses 
    too much RAM. This function quickly gathers metadata and guesses the content type.
    
    Args:
        path (Union[str, Path]): Path to the .npz file.
        
    Returns:
        Dict[str, Any]: Metadata dictionary containing keys, shapes, dtypes, value ranges,
                        and estimated content types per key.
    """
    path = Path(path)
    metadata: Dict[str, Any] = {'file_path': str(path), 'keys': []}
    
    try:
        # Load using mmap_mode='r' to read metadata without keeping full arrays in memory
        with np.load(path, mmap_mode='r', allow_pickle=False) as data:
            keys = list(data.keys())
            metadata['keys'] = keys
            
            for key in keys:
                arr = data[key]
                shape = arr.shape
                dtype = str(arr.dtype)
                
                # To get min/max/unique we need to evaluate the array.
                # If it's too large, this might still take some time, but we process one case at a time.
                min_val = float(np.min(arr))
                max_val = float(np.max(arr))
                unique_vals = np.unique(arr)
                num_unique = len(unique_vals)
                
                # Heuristics to determine content type
                estimated_type = "UNKNOWN — NEEDS VERIFICATION"
                reasoning = []
                
                if dtype == 'uint8':
                    reasoning.append("dtype is uint8")
                
                if num_unique < 20:
                    reasoning.append(f"very few unique values ({num_unique} < 20)")
                    estimated_type = "segmentation labels"
                elif num_unique > 1000:
                    reasoning.append(f"many unique values ({num_unique} > 1000)")
                    estimated_type = "intensity image"
                else:
                    reasoning.append(f"moderate unique values ({num_unique})")
                    
                # Store per-key metadata
                metadata[key] = {
                    'shape': shape,
                    'dtype': dtype,
                    'min_val': min_val,
                    'max_val': max_val,
                    'num_unique': num_unique,
                    'estimated_content_type': estimated_type,
                    'reasoning': ", ".join(reasoning)
                }
                
                print(f"Key '{key}' Analysis:")
                print(f"  Shape: {shape}, Dtype: {dtype}, Range: [{min_val}, {max_val}]")
                print(f"  Estimated Type: {estimated_type}")
                print(f"  Reasoning: {', '.join(reasoning)}")
                print(f"  Caveat: This is a heuristic — verify with dataset documentation.\n")
                
        return metadata
    except Exception as e:
        print(f"Error inspecting {path}: {str(e)}")
        return metadata

def scan_npz_dataset(root_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Recursively scans a directory for .npz files and compiles a summary DataFrame.
    
    Why we need this: We need a bird's-eye view of our entire dataset to ensure
    consistency across files and understand the overall data distribution.
    
    Args:
        root_dir (Union[str, Path]): Root directory to scan.
        
    Returns:
        pd.DataFrame: DataFrame containing metadata for all found arrays.
    """
    root_dir = Path(root_dir)
    npz_files = list(root_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files in {root_dir}")
    
    records = []
    
    for file_path in tqdm(npz_files, desc="Scanning dataset"):
        try:
            # File size in MB
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Inspect metadata
            meta = inspect_npz(file_path)
            
            for key in meta.get('keys', []):
                key_meta = meta.get(key, {})
                records.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'array_key': key,
                    'shape': key_meta.get('shape', "UNKNOWN — NEEDS VERIFICATION"),
                    'dtype': key_meta.get('dtype', "UNKNOWN — NEEDS VERIFICATION"),
                    'min_val': key_meta.get('min_val', "UNKNOWN — NEEDS VERIFICATION"),
                    'max_val': key_meta.get('max_val', "UNKNOWN — NEEDS VERIFICATION"),
                    'num_unique': key_meta.get('num_unique', "UNKNOWN — NEEDS VERIFICATION"),
                    'estimated_content_type': key_meta.get('estimated_content_type', "UNKNOWN — NEEDS VERIFICATION"),
                    'file_size_mb': file_size_mb
                })
        except Exception as e:
            print(f"Skipping {file_path} due to error: {str(e)}")
            
    df = pd.DataFrame(records)
    return df
