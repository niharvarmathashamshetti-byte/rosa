import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.mesh_generator import mask_to_mesh, compute_surface_metrics, LABEL_NAMES

def run_reconstruction(args):
    data_dir = Path(args.data_dir)
    pred_dir = Path(args.predictions_dir)
    out_mesh_dir = Path(args.results_dir) / 'meshes'
    out_mesh_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_split) as f:
        test_ids = [l.strip() for l in f if l.strip()]

    if args.max_cases > 0:
        test_ids = test_ids[:args.max_cases]

    print(f'Running 3D Marching Cubes Physical Reconstruction for {len(test_ids)} test cases...', flush=True)

    mesh_records = []

    for cid in test_ids:
        lbl_p = data_dir / 'labelsTr' / f'{cid}.nii.gz'
        pred_p = pred_dir / f'{cid}_pred.nii.gz'

        if not lbl_p.exists() or not pred_p.exists():
            print(f'Skipping {cid} (prediction or label file not ready)', flush=True)
            continue

        nii_lbl = nib.load(str(lbl_p))
        nii_pred = nib.load(str(pred_p))

        gt_mask = nii_lbl.get_fdata().astype(np.uint8)
        pred_mask = nii_pred.get_fdata().astype(np.uint8)
        affine = nii_lbl.affine
        spacing = tuple(float(x) for x in nii_lbl.header.get_zooms())

        case_mesh_dir = out_mesh_dir / cid
        gt_mesh_dir = case_mesh_dir / 'ground_truth'
        pred_mesh_dir = case_mesh_dir / 'predicted'
        gt_mesh_dir.mkdir(parents=True, exist_ok=True)
        pred_mesh_dir.mkdir(parents=True, exist_ok=True)

        for label_id, label_name in LABEL_NAMES.items():
            # 1. Ground truth mesh
            gt_stl_path = gt_mesh_dir / f'{label_name}.stl'
            gt_res = mask_to_mesh(
                mask=gt_mask,
                label=label_id,
                affine=affine,
                spacing=spacing,
                output_path=gt_stl_path,
                step_size=2
            )

            # 2. Predicted mesh
            pred_stl_path = pred_mesh_dir / f'{label_name}.stl'
            pred_res = mask_to_mesh(
                mask=pred_mask,
                label=label_id,
                affine=affine,
                spacing=spacing,
                output_path=pred_stl_path,
                step_size=2
            )

            # 3. Surface-to-surface metrics
            surf_metrics = {'average_surface_distance_mm': None, 'surface_hd95_mm': None, 'surface_hd_max_mm': None}
            if gt_res['status'] == 'success' and pred_res['status'] == 'success':
                try:
                    surf_metrics = compute_surface_metrics(gt_res['mesh_obj'], pred_res['mesh_obj'], num_samples=1500)
                except Exception as e:
                    pass

            gt_verts = gt_res['num_vertices']
            pred_verts = pred_res['num_vertices']
            asd_val = surf_metrics['average_surface_distance_mm']
            hd95_val = surf_metrics['surface_hd95_mm']

            rec = {
                'case_id': cid,
                'structure': label_name,
                'gt_vertices': gt_verts,
                'gt_faces': gt_res['num_faces'],
                'gt_volume_mm3': gt_res['volume_mm3'],
                'pred_vertices': pred_verts,
                'pred_faces': pred_res['num_faces'],
                'pred_volume_mm3': pred_res['volume_mm3'],
                'extents_gt_mm': str([round(x, 1) for x in gt_res['extents_mm']]) if gt_res['extents_mm'] else None,
                'extents_pred_mm': str([round(x, 1) for x in pred_res['extents_mm']]) if pred_res['extents_mm'] else None,
                'asd_mm': asd_val,
                'surface_hd95_mm': hd95_val,
                'gt_stl': str(gt_stl_path),
                'pred_stl': str(pred_stl_path)
            }
            mesh_records.append(rec)
            print(f'[{cid}] {label_name}: GT verts={gt_verts}, Pred verts={pred_verts}, ASD={asd_val}', flush=True)

    df_mesh = pd.DataFrame(mesh_records)
    report_path = out_mesh_dir / 'mesh_reconstruction_report.csv'
    df_mesh.to_csv(report_path, index=False)
    print(f'Saved mesh reconstruction report to {report_path}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/oaizib')
    parser.add_argument('--predictions_dir', type=str, default='predictions')
    parser.add_argument('--test_split', type=str, default='splits/test.txt')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--max_cases', type=int, default=3)
    args = parser.parse_args()
    run_reconstruction(args)
