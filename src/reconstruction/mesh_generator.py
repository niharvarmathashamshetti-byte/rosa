import struct
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
from skimage import measure
from scipy import ndimage
from scipy.spatial import cKDTree
import trimesh

LABEL_NAMES = {
    1: 'femur',
    2: 'femoral_cartilage',
    3: 'tibia',
    4: 'medial_tibial_cartilage',
    5: 'lateral_tibial_cartilage'
}

def clean_binary_mask(binary_mask: np.ndarray, min_component_size: int = 50) -> np.ndarray:
    if binary_mask.sum() == 0:
        return binary_mask
    labeled_array, num_features = ndimage.label(binary_mask)
    if num_features <= 1:
        return binary_mask
    component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
    cleaned = np.zeros_like(binary_mask)
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_component_size:
            cleaned[labeled_array == i] = 1
    if cleaned.sum() == 0 and len(component_sizes) > 0:
        largest_id = np.argmax(component_sizes) + 1
        cleaned[labeled_array == largest_id] = 1
    return cleaned

def mask_to_mesh(
    mask: np.ndarray,
    label: int,
    affine: Optional[np.ndarray] = None,
    spacing: tuple = (1.0, 1.0, 1.0),
    output_path: Optional[Union[str, Path]] = None,
    step_size: int = 2,
    min_component_size: int = 50
) -> Dict[str, Any]:
    binary = (mask == label).astype(np.uint8)
    binary = clean_binary_mask(binary, min_component_size=min_component_size)
    voxel_count = int(np.sum(binary))

    if voxel_count == 0:
        return {
            'status': 'error',
            'label': label,
            'label_name': LABEL_NAMES.get(label, f'label_{label}'),
            'num_vertices': 0,
            'num_faces': 0,
            'volume_mm3': 0.0,
            'bounds_min': None,
            'bounds_max': None,
            'extents_mm': None,
            'output_path': None,
            'message': f'Label {label} has no voxels.'
        }

    try:
        verts, faces, normals, values = measure.marching_cubes(
            binary,
            level=0.5,
            step_size=step_size
        )

        if affine is not None:
            homog_verts = np.c_[verts, np.ones(len(verts))]
            phys_verts = (affine @ homog_verts.T).T[:, :3]
        else:
            phys_verts = verts * np.array(spacing)

        mesh = trimesh.Trimesh(vertices=phys_verts, faces=faces, process=True)

        bounds_min = mesh.bounds[0].tolist()
        bounds_max = mesh.bounds[1].tolist()
        extents_mm = mesh.extents.tolist()
        mesh_vol = float(abs(mesh.volume)) if mesh.is_watertight else float(voxel_count * np.prod(spacing))

        out_str = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(str(output_path), file_type='stl')
            out_str = str(output_path)

        return {
            'status': 'success',
            'label': label,
            'label_name': LABEL_NAMES.get(label, f'label_{label}'),
            'num_vertices': int(len(mesh.vertices)),
            'num_faces': int(len(mesh.faces)),
            'volume_mm3': mesh_vol,
            'bounds_min': bounds_min,
            'bounds_max': bounds_max,
            'extents_mm': extents_mm,
            'output_path': out_str,
            'mesh_obj': mesh,
            'message': f'Successfully reconstructed {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'label': label,
            'label_name': LABEL_NAMES.get(label, f'label_{label}'),
            'num_vertices': 0,
            'num_faces': 0,
            'volume_mm3': 0.0,
            'bounds_min': None,
            'bounds_max': None,
            'extents_mm': None,
            'output_path': None,
            'message': f'Marching Cubes failed: {str(e)}'
        }

def generate_all_meshes(
    mask: np.ndarray,
    output_dir: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    spacing: tuple = (0.7, 0.3646, 0.3646),
    prefix: str = '',
    step_size: int = 2
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    labels = sorted(int(l) for l in np.unique(mask) if l > 0)
    for lbl in labels:
        name = LABEL_NAMES.get(lbl, f'label_{lbl}')
        fname = f'{prefix}{name}.stl' if prefix else f'{name}.stl'
        stl_path = output_dir / fname
        res = mask_to_mesh(
            mask=mask,
            label=lbl,
            affine=affine,
            spacing=spacing,
            output_path=stl_path,
            step_size=step_size
        )
        results.append(res)
    return results

def compute_surface_metrics(mesh_gt: trimesh.Trimesh, mesh_pred: trimesh.Trimesh, num_samples: int = 2000) -> Dict[str, float]:
    if len(mesh_gt.vertices) == 0 or len(mesh_pred.vertices) == 0:
        return {'average_surface_distance_mm': None, 'surface_hd95_mm': None, 'surface_hd_max_mm': None}
        
    n_gt = min(num_samples, len(mesh_gt.vertices))
    n_pred = min(num_samples, len(mesh_pred.vertices))
    
    pts_gt, _ = trimesh.sample.sample_surface(mesh_gt, n_gt)
    pts_pred, _ = trimesh.sample.sample_surface(mesh_pred, n_pred)

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    d_pred_to_gt, _ = tree_gt.query(pts_pred)
    d_gt_to_pred, _ = tree_pred.query(pts_gt)

    asd = float((np.mean(d_pred_to_gt) + np.mean(d_gt_to_pred)) / 2.0)
    all_dists = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    hd95 = float(np.percentile(all_dists, 95))
    hd_max = float(np.max(all_dists))

    return {
        'average_surface_distance_mm': asd,
        'surface_hd95_mm': hd95,
        'surface_hd_max_mm': hd_max
    }
