import numpy as np
from scipy.ndimage import distance_transform_edt

LABEL_NAMES = {
    0: 'Background',
    1: 'Femur',
    2: 'Femoral Cartilage',
    3: 'Tibia',
    4: 'Medial Tibial Cartilage',
    5: 'Lateral Tibial Cartilage'
}

def compute_per_class_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing: tuple = (0.7, 0.3646, 0.3646), num_classes: int = 6):
    metrics = {}
    fg_dices = []
    fg_ious = []
    
    for c in range(1, num_classes):
        p_c = (pred_mask == c)
        g_c = (gt_mask == c)
        
        intersection = np.logical_and(p_c, g_c).sum()
        union = np.logical_or(p_c, g_c).sum()
        p_sum = p_c.sum()
        g_sum = g_c.sum()
        
        # Dice
        dice = (2.0 * intersection) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0 if (p_sum == 0 and g_sum == 0) else 0.0
        # IoU
        iou = intersection / union if union > 0 else 1.0 if (p_sum == 0 and g_sum == 0) else 0.0
        # Precision
        precision = intersection / p_sum if p_sum > 0 else 1.0 if g_sum == 0 else 0.0
        # Recall
        recall = intersection / g_sum if g_sum > 0 else 1.0 if p_sum == 0 else 0.0
        
        # HD95 in physical mm
        hd95 = compute_hd95(p_c, g_c, spacing=spacing)
        
        name = LABEL_NAMES.get(c, f'Class_{c}')
        metrics[name] = {
            'dice': float(dice),
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'hd95_mm': float(hd95) if hd95 is not None else None,
            'gt_voxels': int(g_sum),
            'pred_voxels': int(p_sum)
        }
        fg_dices.append(dice)
        fg_ious.append(iou)
        
    metrics['mean_foreground_dice'] = float(np.mean(fg_dices))
    metrics['mean_foreground_iou'] = float(np.mean(fg_ious))
    return metrics

def compute_hd95(pred_binary: np.ndarray, gt_binary: np.ndarray, spacing: tuple = (0.7, 0.3646, 0.3646)):
    if pred_binary.sum() == 0 and gt_binary.sum() == 0:
        return 0.0
    if pred_binary.sum() == 0 or gt_binary.sum() == 0:
        return None
        
    # Boundary points
    max_sp = max(spacing)
    dt_gt_in = distance_transform_edt(gt_binary, sampling=spacing)
    dt_pred_in = distance_transform_edt(pred_binary, sampling=spacing)
    
    pred_border = np.logical_and(pred_binary, dt_pred_in <= max_sp)
    gt_border = np.logical_and(gt_binary, dt_gt_in <= max_sp)
    
    if pred_border.sum() == 0 or gt_border.sum() == 0:
        return None
        
    dt_gt_out = distance_transform_edt(~gt_binary, sampling=spacing)
    dt_pred_out = distance_transform_edt(~pred_binary, sampling=spacing)
    
    distances_pred_to_gt = dt_gt_out[pred_border]
    distances_gt_to_pred = dt_pred_out[gt_border]
    
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    return float(np.percentile(all_distances, 95))
