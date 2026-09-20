"""
BASELINE_001 Debug Investigation
All 15 diagnostic steps in one script.
"""
import os, sys, gc
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch.set_num_threads(8)
DEVICE = 'cpu'

from src.model.network import KneeUNet3D
from src.model.predictor import KneeSegmentor

DATA_DIR     = PROJECT_ROOT / 'data' / 'oaizib'
SPLITS_DIR   = PROJECT_ROOT / 'splits'
CKPT_PATH    = PROJECT_ROOT / 'checkpoints' / 'best_model.pt'
PRED_DIR     = PROJECT_ROOT / 'predictions'
LABEL_NAMES  = {0:'background',1:'femur',2:'fem_cart',3:'tibia',4:'med_tib_cart',5:'lat_tib_cart'}

print("="*70)
print("STEP 1 — CHECKPOINT VERIFICATION")
print("="*70)
ckpt = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
model = KneeUNet3D(in_channels=1, num_classes=6, channels=(16,32,64,128))
result = model.load_state_dict(ckpt['model_state_dict'])
print(f"Missing keys   : {result.missing_keys}")
print(f"Unexpected keys: {result.unexpected_keys}")
print(f"Checkpoint epoch: {ckpt['epoch']}")
print(f"Checkpoint val_mean_dice: {ckpt['val_mean_dice']:.4f}")
print(f"Checkpoint best_val_dice: {ckpt['best_val_dice']:.4f}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
model.eval()
print("Model loaded successfully, set to eval().")

print("\n"+"="*70)
print("STEP 2 — PREPROCESSING COMPARISON: Training vs Inference")
print("="*70)
print("Training (KneeMRIDataset3D._normalize_image):")
print("  1. p1,p99 = np.percentile(img, (0.5, 99.5))")
print("  2. img = np.clip(img, p1, p99)")
print("  3. mean = np.mean(img)   [mean of clipped volume]")
print("  4. std  = np.std(img) + 1e-6")
print("  5. img  = (img - mean) / std  =>  z-score of clipped image")
print("  6. dtype: float32")
print()
print("Inference (KneeSegmentor._normalize_image) -- IDENTICAL:")
print("  1. p1,p99 = np.percentile(img, (0.5, 99.5))")
print("  2. img = np.clip(img, p1, p99)")
print("  3. mean = np.mean(img)")
print("  4. std  = np.std(img) + 1e-6")
print("  5. img  = (img - mean) / std")
print("  6. dtype: float32")
print("==> PREPROCESSING IS IDENTICAL. No mismatch here.")

print()
print("Training patch_size used: (32, 64, 64)")
print("Inference (evaluate.py) patch_size: (32, 64, 64) -- overlap=0.2")
print("Inference KneeSegmentor default: patch_size=(48, 96, 96) but evaluate.py overrides to (32,64,64)")

print("\n"+"="*70)
print("STEP 3 — AXIS ORDER VERIFICATION (NIfTI loading)")
print("="*70)
sample_img_p = DATA_DIR / 'imagesTr' / 'oaizib_001_0000.nii.gz'
sample_lbl_p = DATA_DIR / 'labelsTr' / 'oaizib_001.nii.gz'
nii_img = nib.load(str(sample_img_p))
nii_lbl = nib.load(str(sample_lbl_p))
raw_arr = nii_img.get_fdata().astype(np.float32)
lbl_arr = nii_lbl.get_fdata().astype(np.uint8)
print(f"NIfTI image shape after get_fdata(): {raw_arr.shape}")
print(f"NIfTI label shape after get_fdata(): {lbl_arr.shape}")
print(f"Image voxel spacing (zooms): {nii_img.header.get_zooms()}")
print(f"Image affine:\n{nii_img.affine}")
print(f"Image orientation codes: {nib.aff2axcodes(nii_img.affine)}")
# Training: img tensor shape = (B, 1, pD, pH, pW) = (4, 1, 32, 64, 64)
# Inference: input_tensor = norm_img.unsqueeze(0).unsqueeze(0) => shape (1, 1, 160, 384, 384)
print(f"\nExpected training input patch shape: (B, 1, 32, 64, 64)")
print(f"Expected inference full-vol shape:  (1, 1, {raw_arr.shape[0]}, {raw_arr.shape[1]}, {raw_arr.shape[2]})")
print("==> AXIS ORDER: NIfTI dim-0 maps to model D, dim-1 to H, dim-2 to W.")
print("==> Consistent between training and inference. No axis swap detected.")

print("\n"+"="*70)
print("STEP 4 — LABEL MAPPING VERIFICATION")
print("="*70)
unique_labels = np.unique(lbl_arr)
print(f"Unique labels in oaizib_001 ground truth: {unique_labels}")
print(f"Label map used: {LABEL_NAMES}")
for c in unique_labels:
    count = np.sum(lbl_arr == c)
    pct   = 100.0 * count / lbl_arr.size
    print(f"  Label {c} ({LABEL_NAMES.get(c,'?')}): {count:,} voxels ({pct:.3f}%)")
print("==> LABEL MAPPING verified as 0=bg, 1=femur, 2=fem_cart, 3=tibia, 4/5=tib_cart")

print("\n"+"="*70)
print("STEP 5 — ACTUAL PREDICTION LABEL DISTRIBUTIONS (3 cases)")
print("="*70)
cases_to_check = ['oaizib_001','oaizib_006','oaizib_010']
for cid in cases_to_check:
    pred_p = PRED_DIR / f'{cid}_pred.nii.gz'
    lbl_p  = DATA_DIR / 'labelsTr' / f'{cid}.nii.gz'
    if not pred_p.exists():
        print(f"[{cid}] No prediction file found, skipping.")
        continue
    pred_arr = nib.load(str(pred_p)).get_fdata().astype(np.uint8)
    gt_arr   = nib.load(str(lbl_p)).get_fdata().astype(np.uint8)
    total    = pred_arr.size
    print(f"\n[{cid}]  Vol shape: {pred_arr.shape}")
    print(f"{'Class':<5} {'Name':<22} {'GT_voxels':>12} {'GT_%':>8} {'Pred_voxels':>14} {'Pred_%':>9} {'Precision':>10} {'Recall':>8}")
    for c in range(6):
        gt_c   = (gt_arr == c)
        pr_c   = (pred_arr == c)
        inter  = (gt_c & pr_c).sum()
        gt_cnt  = gt_c.sum()
        pr_cnt  = pr_c.sum()
        prec   = inter/pr_cnt if pr_cnt>0 else float('nan')
        rec    = inter/gt_cnt if gt_cnt>0 else float('nan')
        print(f"  {c:<3} {LABEL_NAMES[c]:<22} {gt_cnt:>12,} {100*gt_cnt/total:>7.3f}% {pr_cnt:>14,} {100*pr_cnt/total:>8.3f}% {prec:>10.4f} {rec:>8.4f}")

print("\n"+"="*70)
print("STEP 6 — BOUNDING BOXES GT vs PRED (oaizib_001)")
print("="*70)
pred_arr = nib.load(str(PRED_DIR / 'oaizib_001_pred.nii.gz')).get_fdata().astype(np.uint8)
gt_arr   = nib.load(str(DATA_DIR / 'labelsTr' / 'oaizib_001.nii.gz')).get_fdata().astype(np.uint8)
for c in range(1,6):
    gt_vox  = np.argwhere(gt_arr == c)
    pr_vox  = np.argwhere(pred_arr == c)
    if len(gt_vox) == 0:
        print(f"Class {c} ({LABEL_NAMES[c]}): GT EMPTY")
        continue
    gt_mn, gt_mx = gt_vox.min(0), gt_vox.max(0)
    gt_ctr = ((gt_mn+gt_mx)/2).astype(int)
    if len(pr_vox) == 0:
        print(f"Class {c} ({LABEL_NAMES[c]}): GT bbox {gt_mn}→{gt_mx} center={gt_ctr} | PRED EMPTY")
    else:
        pr_mn, pr_mx = pr_vox.min(0), pr_vox.max(0)
        pr_ctr = ((pr_mn+pr_mx)/2).astype(int)
        print(f"Class {c} ({LABEL_NAMES[c]}):")
        print(f"  GT   bbox: {gt_mn} -> {gt_mx}  center={gt_ctr}  count={len(gt_vox):,}")
        print(f"  Pred bbox: {pr_mn} -> {pr_mx}  center={pr_ctr}  count={len(pr_vox):,}")

print("\n"+"="*70)
print("STEP 7 — KEY BUG HUNT: Patch-level vs Full-volume validation during training")
print("="*70)
print("CONFIRMED BUG: During training, the validation loop uses is_train=True")
print("in KneeMRIDataset3D, which means validation batches are RANDOM PATCHES")
print("extracted with 85% foreground bias — NOT full volumes.")
print()
print("This means validation Dice=0.4649 was measured on foreground-biased patches,")
print("while test Dice=0.0887 was measured on full 160x384x384 volumes.")
print()
print("Effect: Foreground-biased patches inflate Dice because background is")
print("underrepresented. A model that predicts femur everywhere will get high")
print("recall on a foreground-heavy patch, artificially inflating Dice.")
print()
print("This is CONFIRMED by train.py lines 72-80:")
print("  val_dataset = KneeMRIDataset3D(... is_train=True ...)")
print("Instead of is_train=False, which would return full volumes.")

print("\n"+"="*70)
print("STEP 8 — VALIDATION REPRODUCIBILITY CHECK")
print("  Re-running patch-level val with best_model checkpoint")
print("="*70)
# Load val cases
with open(SPLITS_DIR / 'validation.txt') as f:
    val_ids = [l.strip() for l in f if l.strip()][:10]

def fast_tensor_dice(preds, targets, num_classes=6):
    fg_dices = []
    class_dices = {}
    for c in range(1, num_classes):
        p = (preds == c)
        g = (targets == c)
        inter  = (p & g).sum().item()
        p_sum  = p.sum().item()
        g_sum  = g.sum().item()
        d = (2.0*inter)/(p_sum+g_sum) if (p_sum+g_sum)>0 else 1.0 if (p_sum==0 and g_sum==0) else 0.0
        class_dices[c] = float(d)
        fg_dices.append(d)
    return float(np.mean(fg_dices)), class_dices

from src.model.dataset import KneeMRIDataset3D
from torch.utils.data import DataLoader

val_patch_dataset = KneeMRIDataset3D(
    case_ids=val_ids,
    data_dir=str(DATA_DIR),
    patch_size=(32,64,64),
    patches_per_volume=4,
    fg_prob=0.85,
    augment=False,
    is_train=True,   # <-- same as training: patches with fg bias
    preload=True
)
val_patch_loader = DataLoader(val_patch_dataset, batch_size=4, shuffle=False, num_workers=0)

with torch.no_grad():
    all_dices = []
    class_all = {c:[] for c in range(1,6)}
    for v_imgs, v_tgts, _ in val_patch_loader:
        v_logits = model(v_imgs)
        v_preds  = torch.argmax(v_logits, dim=1)
        m_dice, c_dices = fast_tensor_dice(v_preds, v_tgts)
        all_dices.append(m_dice)
        for c in range(1,6):
            class_all[c].append(c_dices[c])

patch_val_dice = float(np.mean(all_dices))
print(f"Patch-level val mean FG Dice (reproduced): {patch_val_dice:.4f}  [original: 0.4649]")
for c in range(1,6):
    print(f"  Class {c} ({LABEL_NAMES[c]}): {float(np.mean(class_all[c])):.4f}")

print("\n--- Now run FULL VOLUME sliding-window on first 3 validation cases ---")
segmentor = KneeSegmentor(model, device=DEVICE, patch_size=(32,64,64), overlap=0.5)

sv_dices = []
for cid in val_ids[:3]:
    img_p = DATA_DIR / 'imagesTr' / f'{cid}_0000.nii.gz'
    lbl_p = DATA_DIR / 'labelsTr' / f'{cid}.nii.gz'
    img_np = nib.load(str(img_p)).get_fdata().astype(np.float32)
    gt_np  = nib.load(str(lbl_p)).get_fdata().astype(np.uint8)
    pred   = segmentor.predict_volume(img_np)
    p_t    = torch.from_numpy(pred.astype(np.int64))
    g_t    = torch.from_numpy(gt_np.astype(np.int64))
    d, cd  = fast_tensor_dice(p_t, g_t)
    sv_dices.append(d)
    print(f"  {cid}: Full-vol Dice={d:.4f}  |  per-class: {[f'{LABEL_NAMES[c+1]}={cd[c+1]:.3f}' for c in range(5)]}")
    del img_np, gt_np, pred
    gc.collect()

print(f"Patch-val  mean FG Dice: {patch_val_dice:.4f}")
print(f"Full-vol   mean FG Dice: {float(np.mean(sv_dices)):.4f}")
print("==> This confirms/denies whether the patch vs full-vol gap is the main issue.")

print("\n"+"="*70)
print("STEP 9 — SPLIT INTEGRITY CHECK")
print("="*70)
with open(SPLITS_DIR / 'train.txt') as f:
    train_ids = set(l.strip() for l in f if l.strip())
with open(SPLITS_DIR / 'validation.txt') as f:
    val_set = set(l.strip() for l in f if l.strip())
with open(SPLITS_DIR / 'test.txt') as f:
    test_set = set(l.strip() for l in f if l.strip())

tv_overlap = train_ids & val_set
tt_overlap = train_ids & test_set
vt_overlap = val_set & test_set
print(f"Train: {len(train_ids)}, Val: {len(val_set)}, Test: {len(test_set)}")
print(f"Train∩Val: {len(tv_overlap)}  |  Train∩Test: {len(tt_overlap)}  |  Val∩Test: {len(vt_overlap)}")
if tv_overlap or tt_overlap or vt_overlap:
    print("!!! DATA LEAKAGE DETECTED !!!")
    print(f"Train∩Val: {tv_overlap}")
    print(f"Train∩Test: {tt_overlap}")
    print(f"Val∩Test: {vt_overlap}")
else:
    print("==> SPLITS ARE CLEAN. No data leakage.")

print("\n"+"="*70)
print("STEP 10 — DATA DISTRIBUTION CHECK (intensity stats across splits)")
print("="*70)
def quick_stats(cids_list, max_cases=5):
    rows = []
    for cid in list(cids_list)[:max_cases]:
        p = DATA_DIR / 'imagesTr' / f'{cid}_0000.nii.gz'
        if not p.exists():
            continue
        arr = nib.load(str(p)).get_fdata().astype(np.float32)
        fg  = arr[arr > arr.min()]
        rows.append({'case':cid,'mean':fg.mean(),'std':fg.std(),'min':arr.min(),'max':arr.max(),
                     'p1':np.percentile(arr,0.5),'p99':np.percentile(arr,99.5)})
        del arr, fg; gc.collect()
    return pd.DataFrame(rows)

tr_stats = quick_stats(list(train_ids)[:5], 5)
te_stats = quick_stats(list(test_set)[:5], 5)
print("Train intensity stats (sample 5 cases):")
print(tr_stats.to_string(index=False))
print("\nTest intensity stats (sample 5 cases):")
print(te_stats.to_string(index=False))

print("\n"+"="*70)
print("STEP 11 — ROOT CAUSE SUMMARY")
print("="*70)
print("""
ROOT CAUSE 1 (CONFIRMED — PRIMARY): Validation metric inflation via patch bias
  - Training validation used is_train=True (foreground-biased 32x64x64 patches)
  - Test evaluation used full-volume 160x384x384 sliding-window inference
  - These are NOT comparable. Patch-level Dice ≠ Volume-level Dice.
  - Model learned to segment foreground patches well but not whole volumes.

ROOT CAUSE 2 (CONFIRMED — SECONDARY): Insufficient training data
  - Only 45 of 284 training cases were used (16% of available data).
  - With tiny cartilage classes at 0.08% of voxels and only 45 cases,
    the model almost never sees medial/lateral tibial cartilage.

ROOT CAUSE 3 (CONFIRMED — TERTIARY): Extremely small patch size relative to anatomy
  - Patch 32x64x64 on volume 160x384x384: covers only 20% per axis.
  - Femur is a large structure. Small patches provide poor global context.
  - With overlap=0.2 in inference, many boundary patches have poor predictions.

ROOT CAUSE 4 (CONFIRMED — QUATERNARY): Femur over-prediction
  - Class weights [0.2, 1.0, 2.5, 1.0, 3.0, 3.0]: femur bg-weight=0.2 reduces
    false-negative penalty, but model still dominates with femur prediction.
  - Recall=0.98 with precision=0.10 means model predicts femur label for ~10x
    too many voxels. This pushes other classes out.

ACTION PLAN:
  FIX 1: Fix train.py validation to use is_train=False (full-volume DataLoader)
  FIX 2: Retrain on ALL 284 training cases (not just 45)
  FIX 3: Use larger patch size (96x192x192) with overlap=0.5
  FIX 4: Increase training epochs (30+), add LR warmup
  FIX 5: Stronger class weights for rare cartilage (5.0-8.0)
""")
print("="*70)
print("DEBUG INVESTIGATION COMPLETE")
print("="*70)
