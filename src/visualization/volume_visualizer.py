import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CLASS_COLORS = {
    1: [0.9, 0.2, 0.2, 0.6],  # Femur: Red
    2: [0.2, 0.8, 0.9, 0.8],  # Femoral Cartilage: Cyan
    3: [0.2, 0.8, 0.2, 0.6],  # Tibia: Green
    4: [0.9, 0.8, 0.2, 0.8],  # Medial Tibial Cartilage: Yellow
    5: [0.9, 0.4, 0.9, 0.8]   # Lateral Tibial Cartilage: Magenta
}

CLASS_NAMES = {
    1: 'Femur',
    2: 'Femoral Cartilage',
    3: 'Tibia',
    4: 'Medial Tibial Cartilage',
    5: 'Lateral Tibial Cartilage'
}

def create_color_overlay(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    for c, col in CLASS_COLORS.items():
        idx = (mask == c)
        rgba[idx] = col
    return rgba

def show_slice(slice_arr: np.ndarray, title: str = '', output_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    p1, p99 = np.percentile(slice_arr, (1, 99))
    norm = np.clip((slice_arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    ax.imshow(norm, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def show_three_views(volume: np.ndarray, title: str = '', output_path: str = None):
    D, H, W = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(volume[D//2, :, :], cmap='gray')
    axes[0].set_title('Sagittal')
    axes[0].axis('off')
    axes[1].imshow(volume[:, H//2, :], cmap='gray')
    axes[1].set_title('Coronal')
    axes[1].axis('off')
    axes[2].imshow(volume[:, :, W//2], cmap='gray')
    axes[2].set_title('Axial')
    axes[2].axis('off')
    fig.suptitle(title)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def show_slice_overlay(slice_img: np.ndarray, slice_mask: np.ndarray, title: str = '', output_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    p1, p99 = np.percentile(slice_img, (1, 99))
    norm = np.clip((slice_img - p1) / (p99 - p1 + 1e-6), 0, 1)
    ax.imshow(norm, cmap='gray')
    overlay = create_color_overlay(slice_mask)
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis('off')
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def plot_case_comparison(
    image_vol: np.ndarray,
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    case_id: str,
    output_path: str,
    slices: dict = None
):
    D, H, W = image_vol.shape
    
    if slices is None:
        sag_idx = int(np.argmax(np.sum(gt_vol > 0, axis=(1, 2))))
        cor_idx = int(np.argmax(np.sum(gt_vol > 0, axis=(0, 2))))
        axi_idx = int(np.argmax(np.sum(gt_vol > 0, axis=(0, 1))))
        
        sag_idx = sag_idx if np.sum(gt_vol[sag_idx] > 0) > 0 else D // 2
        cor_idx = cor_idx if np.sum(gt_vol[:, cor_idx, :] > 0) > 0 else H // 2
        axi_idx = axi_idx if np.sum(gt_vol[:, :, axi_idx] > 0) > 0 else W // 2
    else:
        sag_idx = slices.get('sagittal', D // 2)
        cor_idx = slices.get('coronal', H // 2)
        axi_idx = slices.get('axial', W // 2)

    planes = [
        ('Sagittal (Slice ' + str(sag_idx) + ')', image_vol[sag_idx, :, :], gt_vol[sag_idx, :, :], pred_vol[sag_idx, :, :]),
        ('Coronal (Slice ' + str(cor_idx) + ')', image_vol[:, cor_idx, :], gt_vol[:, cor_idx, :], pred_vol[:, cor_idx, :]),
        ('Axial (Slice ' + str(axi_idx) + ')', image_vol[:, :, axi_idx], gt_vol[:, :, axi_idx], pred_vol[:, :, axi_idx])
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 14), dpi=150)
    plt.subplots_adjust(wspace=0.08, hspace=0.15)

    for row, (plane_name, img_slice, gt_slice, pred_slice) in enumerate(planes):
        p1, p99 = np.percentile(img_slice, (1, 99))
        norm_slice = np.clip((img_slice - p1) / (p99 - p1 + 1e-6), 0, 1)
        
        # 1. Raw MRI
        axes[row, 0].imshow(norm_slice, cmap='gray', aspect='auto')
        axes[row, 0].set_title(f'{plane_name}\nRaw MRI', fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # 2. Ground Truth Overlay
        axes[row, 1].imshow(norm_slice, cmap='gray', aspect='auto')
        gt_overlay = create_color_overlay(gt_slice)
        axes[row, 1].imshow(gt_overlay, aspect='auto')
        axes[row, 1].set_title(f'{plane_name}\nGround Truth Overlay', fontsize=11, fontweight='bold')
        axes[row, 1].axis('off')

        # 3. Prediction Overlay
        axes[row, 2].imshow(norm_slice, cmap='gray', aspect='auto')
        pred_overlay = create_color_overlay(pred_slice)
        axes[row, 2].imshow(pred_overlay, aspect='auto')
        axes[row, 2].set_title(f'{plane_name}\nPrediction Overlay', fontsize=11, fontweight='bold')
        axes[row, 2].axis('off')

        # 4. Error Map
        error_rgb = np.zeros((*img_slice.shape, 3), dtype=np.float32)
        tp = np.logical_and(pred_slice > 0, pred_slice == gt_slice)
        fp = np.logical_and(pred_slice > 0, pred_slice != gt_slice)
        fn = np.logical_and(gt_slice > 0, pred_slice != gt_slice)

        error_rgb[tp] = [0.2, 0.8, 0.2]  # Green = TP
        error_rgb[fp] = [0.9, 0.2, 0.2]  # Red = FP
        error_rgb[fn] = [0.2, 0.4, 0.9]  # Blue = FN

        axes[row, 3].imshow(norm_slice, cmap='gray', aspect='auto')
        axes[row, 3].imshow(error_rgb, alpha=0.65, aspect='auto')
        axes[row, 3].set_title(f'{plane_name}\nError Map (G:TP, R:FP, B:FN)', fontsize=11, fontweight='bold')
        axes[row, 3].axis('off')

    legend_patches = [mpatches.Patch(color=col[:3], label=name) for name, col in [
        ('Femur', [0.9, 0.2, 0.2]),
        ('Femoral Cartilage', [0.2, 0.8, 0.9]),
        ('Tibia', [0.2, 0.8, 0.2]),
        ('Medial Tib Cartilage', [0.9, 0.8, 0.2]),
        ('Lateral Tib Cartilage', [0.9, 0.4, 0.9])
    ]]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f'Case: {case_id} — 3D Anatomical Segmentation Verification', fontsize=15, fontweight='bold', y=0.98)
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), bbox_inches='tight', dpi=150)
    plt.close(fig)
