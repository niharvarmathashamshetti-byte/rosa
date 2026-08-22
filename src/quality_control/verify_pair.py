"""
=============================================================================
ROSA Knee AI — Comprehensive Pair Verification Script
=============================================================================
Verifies one complete CT/MR image + segmentation-mask pair:
  1. Metadata inspection (Shape, Spacing, Origin, Direction Matrix)
  2. Spatial alignment verification (Exact coordinate match)
  3. Label distribution & analysis (Unique labels, voxel counts, volume %)
  4. Orthogonal slice visualization (Axial, Coronal, Sagittal)
  5. Multi-panel overlay figure generation (Image, Mask, Overlay)
  6. Saves high-res verification figure to outputs/figures/
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def verify_image_mask_pair(
    image_path: Path,
    mask_path: Path,
    output_figure_path: Path,
    case_name: str = "oaizib_005"
) -> dict:
    print("=" * 70)
    print(f"  IMAGE + MASK TRAINING PAIR VERIFICATION: {case_name}")
    print("=" * 70)

    # 1. Load with SimpleITK to preserve spatial metadata
    print(f"\n[1] Loading image: {image_path.name}")
    sitk_img = sitk.ReadImage(str(image_path))
    
    print(f"[2] Loading mask:  {mask_path.name}")
    sitk_mask = sitk.ReadImage(str(mask_path))

    # Extract spatial parameters
    img_size = sitk_img.GetSize()        # (X, Y, Z)
    mask_size = sitk_mask.GetSize()
    
    img_spacing = sitk_img.GetSpacing()  # (sx, sy, sz)
    mask_spacing = sitk_mask.GetSpacing()
    
    img_origin = sitk_img.GetOrigin()    # (ox, oy, oz)
    mask_origin = sitk_mask.GetOrigin()
    
    img_direction = sitk_img.GetDirection()
    mask_direction = sitk_mask.GetDirection()

    print("\n--- SPATIAL METADATA COMPARISON ---")
    print(f"Matrix Dimensions (X, Y, Z):")
    print(f"  Image: {img_size}")
    print(f"  Mask:  {mask_size}")
    shape_match = (img_size == mask_size)
    print(f"  --> MATCH: {'YES [PASS]' if shape_match else 'NO [FAIL]'}")

    print(f"\nVoxel Spacing (dx, dy, dz) mm:")
    print(f"  Image: {[round(s, 4) for s in img_spacing]}")
    print(f"  Mask:  {[round(s, 4) for s in mask_spacing]}")
    spacing_match = np.allclose(img_spacing, mask_spacing, atol=1e-4)
    print(f"  --> MATCH: {'YES [PASS]' if spacing_match else 'NO [FAIL]'}")

    print(f"\nPhysical Origin (ox, oy, oz) mm:")
    print(f"  Image: {[round(o, 2) for o in img_origin]}")
    print(f"  Mask:  {[round(o, 2) for o in mask_origin]}")
    origin_match = np.allclose(img_origin, mask_origin, atol=1e-4)
    print(f"  --> MATCH: {'YES [PASS]' if origin_match else 'NO [FAIL]'}")

    print(f"\nDirection Cosines Matrix:")
    dir_match = np.allclose(img_direction, mask_direction, atol=1e-4)
    print(f"  --> MATCH: {'YES [PASS]' if dir_match else 'NO [FAIL]'}")

    # Physical Volume Extent
    phys_extent_img = [round(s * sp, 2) for s, sp in zip(img_size, img_spacing)]
    print(f"\nPhysical FOV Size: {phys_extent_img[0]} x {phys_extent_img[1]} x {phys_extent_img[2]} mm")

    # 2. Extract Numpy arrays
    # SimpleITK GetArrayFromImage gives array in (Z, Y, X) order
    img_arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(sitk_mask).astype(np.uint8)

    print("\n--- INTENSITY & LABEL ANALYSIS ---")
    print(f"Image Array Shape (Z, Y, X): {img_arr.shape}")
    print(f"Image Dtype: {img_arr.dtype}, Min: {np.min(img_arr):.1f}, Max: {np.max(img_arr):.1f}, Mean: {np.mean(img_arr):.1f}")
    
    unique_labels, label_counts = np.unique(mask_arr, return_counts=True)
    total_voxels = mask_arr.size
    print(f"Mask Array Shape (Z, Y, X):  {mask_arr.shape}")
    print(f"Unique Labels in Mask: {unique_labels.tolist()}")
    
    print("\nLabel Distribution Table:")
    print(f"{'Label ID':<10} | {'Voxel Count':<14} | {'Volume %':<10} | {'Status/Description'}")
    print("-" * 65)

    # Standard OAI ZIB / Knee segmentation multi-class convention:
    # 0: Background
    # 1: Femoral Bone
    # 2: Femoral Cartilage
    # 3: Tibial Bone
    # 4: Tibial Cartilage
    # 5: Patellar Bone
    # 6: Patellar Cartilage
    # 7: Meniscus / other structure
    label_annotations = {
        0: "Background / Air / Soft Tissue",
        1: "Femur Bone (Distal Epiphysis)",
        2: "Femoral Cartilage",
        3: "Tibia Bone (Proximal Epiphysis)",
        4: "Tibial Cartilage (Medial + Lateral)",
        5: "Patella Bone",
        6: "Patellar Cartilage",
        7: "Meniscus / Ligamentous structure",
    }

    for lbl, cnt in zip(unique_labels, label_counts):
        pct = (cnt / total_voxels) * 100
        desc = label_annotations.get(lbl, "Unlabeled Anatomical Substructure")
        print(f"{lbl:<10} | {cnt:<14,} | {pct:<9.3f}% | {desc}")
    print("-" * 65)

    # 3. Multi-View Overlay Visualization
    print("\n--- GENERATING 3-PLANE OVERLAY VISUALIZATIONS ---")
    # Find middle slices or slices with dense bone/cartilage
    # Let's find the slice along each axis with the maximum number of labeled voxels
    z_vox_counts = np.sum(mask_arr > 0, axis=(1, 2))
    y_vox_counts = np.sum(mask_arr > 0, axis=(0, 2))
    x_vox_counts = np.sum(mask_arr > 0, axis=(0, 1))

    best_z = int(np.argmax(z_vox_counts))
    best_y = int(np.argmax(y_vox_counts))
    best_x = int(np.argmax(x_vox_counts))

    print(f"Selected representative slices with dense knee anatomy:")
    print(f"  Axial Slice (Z):    {best_z} / {img_arr.shape[0]-1}")
    print(f"  Coronal Slice (Y):  {best_y} / {img_arr.shape[1]-1}")
    print(f"  Sagittal Slice (X): {best_x} / {img_arr.shape[2]-1}")

    # Build 3x3 visualization figure:
    # Rows: Axial, Coronal, Sagittal
    # Cols: Medical Image (Grayscale), Segmentation Mask (Categorical), Transparency Overlay
    fig, axes = plt.subplots(3, 3, figsize=(16, 16), dpi=150)
    fig.patch.set_facecolor("#0a0e17")

    views = [
        ("Axial (Z-plane)", 0, best_z, img_arr[best_z, :, :], mask_arr[best_z, :, :]),
        ("Coronal (Y-plane)", 1, best_y, img_arr[:, best_y, :], mask_arr[:, best_y, :]),
        ("Sagittal (X-plane)", 2, best_x, img_arr[:, :, best_x], mask_arr[:, :, best_x]),
    ]

    cmap_mask = plt.get_cmap("tab10", 10)

    for row_idx, (view_name, axis_id, sl_idx, slice_img, slice_mask) in enumerate(views):
        # 1. Image
        ax_img = axes[row_idx, 0]
        ax_img.imshow(slice_img, cmap="gray")
        ax_img.set_title(f"{view_name} — Image (Slice {sl_idx})", color="#fff", fontsize=12, fontweight="bold")
        ax_img.axis("off")

        # 2. Mask
        ax_mask = axes[row_idx, 1]
        ax_mask.imshow(slice_mask, cmap=cmap_mask, vmin=0, vmax=9, interpolation="nearest")
        ax_mask.set_title(f"{view_name} — Segmentation Mask", color="#fff", fontsize=12, fontweight="bold")
        ax_mask.axis("off")

        # 3. Overlay
        ax_ov = axes[row_idx, 2]
        ax_ov.imshow(slice_img, cmap="gray")
        masked_mask = np.ma.masked_where(slice_mask == 0, slice_mask)
        ax_ov.imshow(masked_mask, cmap=cmap_mask, alpha=0.55, vmin=0, vmax=9, interpolation="nearest")
        ax_ov.set_title(f"{view_name} — Anatomical Overlay", color="#60a5fa", fontsize=12, fontweight="bold")
        ax_ov.axis("off")

    # Legend for labels
    legend_patches = [
        mpatches.Patch(color=cmap_mask(lbl), label=f"Label {lbl}: {label_annotations.get(lbl, 'Unknown')}")
        for lbl in unique_labels if lbl > 0
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=3,
        fontsize=10,
        facecolor="#111827",
        edgecolor="#374151",
        labelcolor="#f3f4f6",
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.suptitle(
        f"ROSA Knee AI — Training Pair Verification & Spatial Alignment\nCase: {case_name} | FOV: {phys_extent_img[0]}x{phys_extent_img[1]}x{phys_extent_img[2]} mm | Spacing: {[round(s,3) for s in img_spacing]} mm",
        color="#f3f4f6",
        fontsize=14,
        fontweight="bold",
        y=0.98
    )

    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.15, wspace=0.1)
    plt.savefig(str(output_figure_path), facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"\n[OK] High-resolution verification figure saved to:")
    print(f"     {output_figure_path}")
    print("=" * 70)

    return {
        "case_name": case_name,
        "shape_match": shape_match,
        "spacing_match": spacing_match,
        "origin_match": origin_match,
        "dir_match": dir_match,
        "all_match": shape_match and spacing_match and origin_match and dir_match,
        "labels": unique_labels.tolist(),
        "dimensions": list(img_size),
        "spacing": list(img_spacing),
        "figure_path": str(output_figure_path),
    }


if __name__ == "__main__":
    img_p = Path("data/raw/imagesTr/oaizib_005_0000.nii.gz")
    lbl_p = Path("data/raw/labelsTr/oaizib_005.nii.gz")
    fig_out = Path("outputs/figures/pair_verification_oaizib_005.png")

    if not img_p.exists() or not lbl_p.exists():
        print(f"Error: Could not find extracted pair in data/raw/")
        sys.exit(1)

    result = verify_image_mask_pair(img_p, lbl_p, fig_out, case_name="oaizib_005")
