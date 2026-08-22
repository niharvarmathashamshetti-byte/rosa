"""
=============================================================================
ROSA Knee AI — 3D Mesh Generator
=============================================================================
Converts segmentation masks into 3D surface meshes (STL format) using
the Marching Cubes algorithm.

This module can work independently of the AI model:
    - If we have real segmentation masks (from our .npz dataset),
      we can generate STL meshes right now.
    - When the trained model produces masks, this module processes
      them exactly the same way.

Pipeline:
    Segmentation mask (3D numpy array)
        → Marching Cubes (per label)
        → 3D mesh (vertices + faces)
        → STL file (one per structure)

Dependencies:
    - scikit-image for marching_cubes
    - numpy for array operations
    - struct for binary STL writing

IMPORTANT:
    - Do NOT generate fake meshes
    - Only process real segmentation data
    - Label meanings are UNKNOWN until verified with dataset documentation
=============================================================================
"""

import struct
from pathlib import Path
from typing import Optional

import numpy as np
from skimage import measure


def mask_to_mesh(
    mask: np.ndarray,
    label: int,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_path: Optional[Path] = None,
    step_size: int = 1,
) -> dict:
    """
    Convert a single label from a segmentation mask into a 3D mesh.

    Uses the Marching Cubes algorithm to extract an isosurface from the
    binary mask of a specific label.

    Args:
        mask: 3D numpy array containing integer labels (e.g., 0-7).
        label: Which label to extract (e.g., 1 for one structure).
        spacing: Voxel spacing (z, y, x) in mm for physical scaling.
        output_path: If provided, saves the mesh as a binary STL file.
        step_size: Step size for marching cubes (higher = faster but coarser).

    Returns:
        Dictionary with:
            - 'status': 'success' or 'error'
            - 'label': the label that was processed
            - 'num_vertices': number of vertices in the mesh
            - 'num_faces': number of triangles
            - 'output_path': path to saved STL (if saved)
            - 'message': status message
    """
    # Create binary mask for this label
    binary = (mask == label).astype(np.uint8)

    # Check if the label has any voxels
    voxel_count = np.sum(binary)
    if voxel_count == 0:
        return {
            "status": "error",
            "label": label,
            "num_vertices": 0,
            "num_faces": 0,
            "output_path": None,
            "message": f"Label {label} has no voxels in the mask.",
        }

    try:
        # Run Marching Cubes to extract the surface mesh
        # level=0.5 finds the boundary between 0 and 1 in our binary mask
        vertices, faces, normals, values = measure.marching_cubes(
            binary,
            level=0.5,
            spacing=spacing,
            step_size=step_size,
        )

        result = {
            "status": "success",
            "label": label,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
            "output_path": None,
            "message": f"Mesh generated: {len(vertices)} vertices, {len(faces)} faces.",
        }

        # Save as STL if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _write_binary_stl(output_path, vertices, faces, normals)
            result["output_path"] = str(output_path)
            result["message"] += f" Saved to {output_path}"

        return result

    except Exception as e:
        return {
            "status": "error",
            "label": label,
            "num_vertices": 0,
            "num_faces": 0,
            "output_path": None,
            "message": f"Marching cubes failed for label {label}: {str(e)}",
        }


def generate_all_meshes(
    mask: np.ndarray,
    output_dir: Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    label_names: Optional[dict[int, str]] = None,
    step_size: int = 1,
) -> list[dict]:
    """
    Generate STL meshes for all non-background labels in a segmentation mask.

    Args:
        mask: 3D segmentation mask with integer labels.
        output_dir: Directory to save STL files.
        spacing: Voxel spacing (z, y, x) in mm.
        label_names: Optional mapping of label ID to structure name.
                     If not provided, uses "label_N" as filename.
                     DO NOT assume label meanings without verification.
        step_size: Marching cubes step size.

    Returns:
        List of result dictionaries, one per label.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all unique labels, skip background (0)
    labels = sorted(int(l) for l in np.unique(mask) if l > 0)

    if not labels:
        return [{
            "status": "error",
            "label": -1,
            "message": "No non-background labels found in the mask.",
        }]

    results = []
    for label in labels:
        # Determine filename
        if label_names and label in label_names:
            name = label_names[label]
        else:
            # DO NOT assume what the label represents
            name = f"label_{label}"

        stl_path = output_dir / f"{name}.stl"
        result = mask_to_mesh(
            mask=mask,
            label=label,
            spacing=spacing,
            output_path=stl_path,
            step_size=step_size,
        )
        result["structure_name"] = name
        results.append(result)

    return results


def _write_binary_stl(
    filepath: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
) -> None:
    """
    Write a mesh as a binary STL file.

    Binary STL format:
        - 80-byte header
        - 4-byte triangle count
        - For each triangle: 12-byte normal + 3×12-byte vertices + 2-byte attribute
    """
    with open(filepath, "wb") as f:
        # 80-byte header
        header = b"ROSA Knee AI - Generated STL" + b"\0" * 52
        f.write(header[:80])

        # Number of triangles
        f.write(struct.pack("<I", len(faces)))

        # Write each triangle
        for face_idx in range(len(faces)):
            # Face normal (use the precomputed normal from marching cubes)
            if face_idx < len(normals):
                # Average the vertex normals for this face
                face_verts = faces[face_idx]
                n = np.mean(normals[face_verts], axis=0)
                n_norm = np.linalg.norm(n)
                if n_norm > 0:
                    n = n / n_norm
            else:
                n = np.array([0.0, 0.0, 1.0])

            f.write(struct.pack("<3f", *n))

            # Three vertices
            for vert_idx in faces[face_idx]:
                f.write(struct.pack("<3f", *vertices[vert_idx]))

            # Attribute byte count (unused, set to 0)
            f.write(struct.pack("<H", 0))
