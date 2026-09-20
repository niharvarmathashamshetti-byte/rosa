import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.network import KneeUNet3D
from src.model.predictor import KneeSegmentor
from src.model.metrics import compute_per_class_metrics, LABEL_NAMES

def evaluate_80(args):
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Experiment_002 final test evaluation = 80 held-out cases', flush=True)
    print(f'Using device: {device} | threads: {torch.get_num_threads()}', flush=True)

    # Load model
    checkpoint_path = Path(args.checkpoint)
    print(f'Loading model weights from {checkpoint_path}...', flush=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = KneeUNet3D(in_channels=1, num_classes=6, channels=(16, 32, 64, 128), dropout=0.1).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    segmentor = KneeSegmentor(
        model,
        device=device,
        patch_size=(args.patch_d, args.patch_h, args.patch_w),
        overlap=args.overlap
    )

    # Output dirs
    pred_dir = Path(args.predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.results_dir) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Test cases (all 80 cases from test_split)
    with open(args.test_split) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    if args.max_test_cases > 0:
        test_ids = test_ids[:args.max_test_cases]

    print(f'Evaluating complete held-out test split ({len(test_ids)} cases)...', flush=True)

    per_case_records = []
    class_stats = {name: {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'hd95': []} for name in list(LABEL_NAMES.values())[1:]}

    for idx, cid in enumerate(test_ids, 1):
        img_p = Path(args.data_dir) / 'imagesTr' / f'{cid}_0000.nii.gz'
        lbl_p = Path(args.data_dir) / 'labelsTr' / f'{cid}.nii.gz'

        if not img_p.exists() or not lbl_p.exists():
            print(f'Skipping missing test case: {cid}', flush=True)
            continue

        nii_img = nib.load(str(img_p))
        nii_lbl = nib.load(str(lbl_p))

        img_np = nii_img.get_fdata().astype(np.float32)
        gt_np = nii_lbl.get_fdata().astype(np.uint8)
        spacing = tuple(float(x) for x in nii_img.header.get_zooms())

        # Predict
        pred_mask = segmentor.predict_volume(img_np)

        # Save prediction NIfTI (distinguishable/safe output path)
        pred_nii_path = pred_dir / f'{cid}_pred.nii.gz'
        pred_nii = nib.Nifti1Image(pred_mask, affine=nii_img.affine, header=nii_img.header)
        nib.save(pred_nii, str(pred_nii_path))

        # Compute metrics
        case_metrics = compute_per_class_metrics(pred_mask, gt_np, spacing=spacing)
        
        fg_dice = case_metrics['mean_foreground_dice']
        fg_iou = case_metrics['mean_foreground_iou']

        row = {'case_id': cid, 'mean_fg_dice': fg_dice, 'mean_fg_iou': fg_iou}
        for cname in list(LABEL_NAMES.values())[1:]:
            c_data = case_metrics[cname]
            row[f'{cname}_dice'] = c_data['dice']
            row[f'{cname}_iou'] = c_data['iou']
            row[f'{cname}_precision'] = c_data['precision']
            row[f'{cname}_recall'] = c_data['recall']
            row[f'{cname}_hd95_mm'] = c_data['hd95_mm']

            class_stats[cname]['dice'].append(c_data['dice'])
            class_stats[cname]['iou'].append(c_data['iou'])
            class_stats[cname]['precision'].append(c_data['precision'])
            class_stats[cname]['recall'].append(c_data['recall'])
            if c_data['hd95_mm'] is not None:
                class_stats[cname]['hd95'].append(c_data['hd95_mm'])

        per_case_records.append(row)
        print(f'[{idx:02d}/{len(test_ids):02d}] Case {cid}: Mean FG Dice = {fg_dice:.4f}, IoU = {fg_iou:.4f}', flush=True)

    df_cases = pd.DataFrame(per_case_records)
    # Save separately without overwriting existing test_cases_detailed.csv
    cases_out_csv = metrics_dir / 'test_cases_80cases.csv'
    df_cases.to_csv(cases_out_csv, index=False)

    # Summary table with Mean +/- Std and Median
    summary_records = []
    for cname, vals in class_stats.items():
        m_dice, s_dice, med_dice = float(np.mean(vals['dice'])), float(np.std(vals['dice'])), float(np.median(vals['dice']))
        m_iou, s_iou, med_iou = float(np.mean(vals['iou'])), float(np.std(vals['iou'])), float(np.median(vals['iou']))
        m_prec, s_prec, med_prec = float(np.mean(vals['precision'])), float(np.std(vals['precision'])), float(np.median(vals['precision']))
        m_rec, s_rec, med_rec = float(np.mean(vals['recall'])), float(np.std(vals['recall'])), float(np.median(vals['recall']))
        
        if len(vals['hd95']) > 0:
            m_hd, s_hd, med_hd = float(np.mean(vals['hd95'])), float(np.std(vals['hd95'])), float(np.median(vals['hd95']))
            hd_str = f'{m_hd:.2f} +/- {s_hd:.2f}'
            med_hd_str = f'{med_hd:.2f}'
        else:
            hd_str = 'N/A'
            med_hd_str = 'N/A'
        
        summary_records.append({
            'Anatomical Structure': cname,
            'Dice Score (Mean +/- Std)': f'{m_dice:.4f} +/- {s_dice:.4f}',
            'Dice (Median)': f'{med_dice:.4f}',
            'IoU (Mean +/- Std)': f'{m_iou:.4f} +/- {s_iou:.4f}',
            'Precision (Mean +/- Std)': f'{m_prec:.4f} +/- {s_prec:.4f}',
            'Recall (Mean +/- Std)': f'{m_rec:.4f} +/- {s_rec:.4f}',
            'HD95 mm (Mean +/- Std)': hd_str,
            'HD95 mm (Median)': med_hd_str
        })

    m_fg_dice, s_fg_dice, med_fg_dice = float(df_cases['mean_fg_dice'].mean()), float(df_cases['mean_fg_dice'].std()), float(df_cases['mean_fg_dice'].median())
    m_fg_iou, s_fg_iou, med_fg_iou = float(df_cases['mean_fg_iou'].mean()), float(df_cases['mean_fg_iou'].std()), float(df_cases['mean_fg_iou'].median())

    summary_records.append({
        'Anatomical Structure': 'OVERALL MEAN FOREGROUND',
        'Dice Score (Mean +/- Std)': f'{m_fg_dice:.4f} +/- {s_fg_dice:.4f}',
        'Dice (Median)': f'{med_fg_dice:.4f}',
        'IoU (Mean +/- Std)': f'{m_fg_iou:.4f} +/- {s_fg_iou:.4f}',
        'Precision (Mean +/- Std)': '-',
        'Recall (Mean +/- Std)': '-',
        'HD95 mm (Mean +/- Std)': '-',
        'HD95 mm (Median)': '-'
    })

    df_summary = pd.DataFrame(summary_records)
    summary_out_csv = metrics_dir / 'test_metrics_80cases.csv'
    df_summary.to_csv(summary_out_csv, index=False)
    print('\n================ EXPERIMENT_002 FINAL 80-CASE TEST EVALUATION SUMMARY ================', flush=True)
    print(df_summary.to_string(index=False), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='experiments/experiment_002/checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='data/oaizib')
    parser.add_argument('--test_split', type=str, default='splits/test.txt')
    parser.add_argument('--predictions_dir', type=str, default='experiments/experiment_002/predictions_80cases')
    parser.add_argument('--results_dir', type=str, default='experiments/experiment_002/results')
    parser.add_argument('--patch_d', type=int, default=48)
    parser.add_argument('--patch_h', type=int, default=128)
    parser.add_argument('--patch_w', type=int, default=128)
    parser.add_argument('--overlap', type=float, default=0.5)
    parser.add_argument('--max_test_cases', type=int, default=0)
    args = parser.parse_args()
    evaluate_80(args)
