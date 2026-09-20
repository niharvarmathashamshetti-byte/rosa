import os
import sys
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.volume_visualizer import plot_case_comparison

def generate_visualizations(args):
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.predictions_dir)
    vis_dir = Path(args.results_dir) / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_split) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    if args.max_cases > 0:
        test_ids = test_ids[:args.max_cases]

    print(f'Generating multiplanar visualizations for {len(test_ids)} test cases...')

    for cid in test_ids:
        img_p = data_dir / 'imagesTr' / f'{cid}_0000.nii.gz'
        lbl_p = data_dir / 'labelsTr' / f'{cid}.nii.gz'
        pred_p = pred_dir / f'{cid}_pred.nii.gz'

        if not img_p.exists() or not lbl_p.exists() or not pred_p.exists():
            continue

        img_np = nib.load(str(img_p)).get_fdata().astype(np.float32)
        gt_np = nib.load(str(lbl_p)).get_fdata().astype(np.uint8)
        pred_np = nib.load(str(pred_p)).get_fdata().astype(np.uint8)

        out_img_path = vis_dir / f'{cid}_comparison.png'
        plot_case_comparison(
            image_vol=img_np,
            gt_vol=gt_np,
            pred_vol=pred_np,
            case_id=cid,
            output_path=str(out_img_path)
        )
        print(f'Generated visualization: {out_img_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/oaizib')
    parser.add_argument('--predictions_dir', type=str, default='predictions')
    parser.add_argument('--test_split', type=str, default='splits/test.txt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--max_cases', type=int, default=5)
    args = parser.parse_args()
    generate_visualizations(args)
