"""
EXPERIMENT_002: Fixed Training with Epoch-wise Case Rotation
Fixes from BASELINE_001 debug investigation:
1. Full-volume sliding-window validation (NOT patches)
2. Rotates through all 284 training cases across epochs (30 per epoch, in-RAM)
3. Larger patch size: 48x128x128
4. Overlap 0.5 for inference  
5. 30 epochs with linear warmup + cosine LR
6. Stronger class weights for rare cartilage
"""
import os, sys, time, argparse, gc
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.network import KneeUNet3D
from src.model.dataset import KneeMRIDataset3D
from src.model.losses import DiceCELoss
from src.model.predictor import KneeSegmentor


def full_volume_dice(model, val_ids, data_dir, device, patch_size, num_classes=6):
    segmentor = KneeSegmentor(model, device=device, patch_size=patch_size, overlap=0.5)
    data_dir = Path(data_dir)
    fg_dices = []
    class_dices_all = {c: [] for c in range(1, num_classes)}
    model.eval()
    for cid in val_ids:
        img_p = data_dir / 'imagesTr' / f'{cid}_0000.nii.gz'
        lbl_p = data_dir / 'labelsTr' / f'{cid}.nii.gz'
        if not img_p.exists() or not lbl_p.exists():
            continue
        img_np = nib.load(str(img_p)).get_fdata().astype(np.float32)
        gt_np  = nib.load(str(lbl_p)).get_fdata().astype(np.uint8)
        print(f'    Val inference: {cid} ...', flush=True)
        with torch.no_grad():
            pred = segmentor.predict_volume(img_np)
        p_t  = torch.from_numpy(pred.astype(np.int64))
        g_t  = torch.from_numpy(gt_np.astype(np.int64))
        per  = []
        for c in range(1, num_classes):
            p_c   = (p_t == c); g_c = (g_t == c)
            inter = (p_c & g_c).sum().item()
            p_sum = p_c.sum().item(); g_sum = g_c.sum().item()
            d = (2.0*inter)/(p_sum+g_sum) if (p_sum+g_sum) > 0 else (1.0 if g_sum==0 else 0.0)
            per.append(d); class_dices_all[c].append(d)
        fg_dices.append(float(np.mean(per)))
        del img_np, gt_np, pred, p_t, g_t; gc.collect()
    mean_fg = float(np.mean(fg_dices)) if fg_dices else 0.0
    mean_per = {c: float(np.mean(class_dices_all[c])) if class_dices_all[c] else 0.0
                for c in range(1, num_classes)}
    return mean_fg, mean_per


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(args):
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== EXPERIMENT_002 Fixed ROSA Knee Training ===', flush=True)
    print(f'Device: {device} | Threads: {torch.get_num_threads()}', flush=True)

    splits_dir = Path(args.splits_dir)
    ckpt_dir   = Path(args.checkpoints_dir)
    logs_dir   = Path(args.logs_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / 'train.txt') as f:
        all_train = [l.strip() for l in f if l.strip()]
    with open(splits_dir / 'validation.txt') as f:
        all_val   = [l.strip() for l in f if l.strip()]

    # Use ALL training cases via rotation; val uses first N cases
    val_ids = all_val[:args.max_val_cases] if args.max_val_cases > 0 else all_val
    cases_per_epoch = args.cases_per_epoch  # how many cases to preload per epoch
    rng = np.random.RandomState(42)

    patch_size = (args.patch_d, args.patch_h, args.patch_w)
    print(f'Total train cases: {len(all_train)} | Cases per epoch: {cases_per_epoch}', flush=True)
    print(f'Val cases (full-vol): {len(val_ids)} | Patch: {patch_size}', flush=True)
    print(f'Epochs: {args.epochs} | Warmup: {args.warmup_epochs} | Val every: {args.val_every} ep', flush=True)

    # FIXED: stronger class weights, especially for rare cartilage
    class_weights = torch.tensor([0.1, 1.0, 4.0, 1.0, 8.0, 8.0], dtype=torch.float32).to(device)

    model = KneeUNet3D(in_channels=1, num_classes=6, channels=(16,32,64,128), dropout=0.1).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}', flush=True)

    criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5, class_weights=class_weights)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = args.warmup_epochs
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        prog = float(ep - warmup_epochs) / float(max(1, args.epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch   = 1
    best_val_dice = -1.0
    history       = []
    last_ckpt = ckpt_dir / 'last_model.pt'
    best_ckpt = ckpt_dir / 'best_model.pt'
    hist_csv  = logs_dir / 'training_history.csv'

    if args.resume and last_ckpt.exists():
        print(f'Resuming from {last_ckpt}...', flush=True)
        ck = torch.load(str(last_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        scheduler.load_state_dict(ck['scheduler_state_dict'])
        start_epoch   = ck['epoch'] + 1
        best_val_dice = ck.get('best_val_dice', -1.0)
        history = pd.read_csv(hist_csv).to_dict('records') if hist_csv.exists() else []
        print(f'Resumed at epoch {start_epoch}, best val dice: {best_val_dice:.4f}', flush=True)

    total_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Sample cases for this epoch (rotate through full dataset)
        epoch_cases = list(rng.choice(all_train, size=min(cases_per_epoch, len(all_train)), replace=False))
        print(f'\nEp[{epoch:03d}] Loading {len(epoch_cases)} cases...', flush=True)

        train_dataset = KneeMRIDataset3D(
            case_ids=epoch_cases,
            data_dir=args.data_dir,
            patch_size=patch_size,
            patches_per_volume=args.patches_per_volume,
            fg_prob=0.85,
            augment=True,
            is_train=True,
            preload=True  # Always preload since cases_per_epoch <= 30
        )
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        steps_per_ep  = len(train_loader)

        model.train()
        losses = []
        for step, (imgs, tgts, _) in enumerate(train_loader, 1):
            imgs  = imgs.to(device)
            tgts  = tgts.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if step % 20 == 0 or step == steps_per_ep:
                print(f'Ep[{epoch:03d}] Step[{step:03d}/{steps_per_ep:03d}] Loss={loss.item():.4f} LR={get_lr(optimizer):.6f}', flush=True)

        # Clean up dataset to free RAM before potential validation
        del train_dataset, train_loader
        gc.collect()

        scheduler.step()
        train_loss = float(np.mean(losses))

        run_val  = (epoch % args.val_every == 0) or (epoch == args.epochs) or (epoch == 1)
        val_dice = best_val_dice if history else 0.0
        val_per  = {}

        if run_val:
            print(f'  [Val Ep{epoch:03d}] Full-volume inference on {len(val_ids)} cases...', flush=True)
            val_dice, val_per = full_volume_dice(model, val_ids, args.data_dir, device, patch_size)
            lnames = {1:'femur', 2:'fem_cart', 3:'tibia', 4:'med_tib', 5:'lat_tib'}
            per_str = ' | '.join([f"{lnames[c]}={val_per.get(c,0):.3f}" for c in range(1,6)])
            print(f'  [Val] Full-vol mean FG Dice: {val_dice:.4f}  [{per_str}]', flush=True)

        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice

        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_mean_dice': val_dice,
            'val_per_class': val_per,
            'best_val_dice': best_val_dice,
            'config': vars(args),
        }
        torch.save(ckpt_state, str(last_ckpt))
        if is_best and run_val:
            torch.save(ckpt_state, str(best_ckpt))
            print(f'  *** [BEST] Ep{epoch:03d} Full-Vol Val Dice: {best_val_dice:.4f} ***', flush=True)

        ep_time = time.time() - t0
        rec = {
            'epoch': epoch, 'train_loss': train_loss, 'val_mean_dice': val_dice,
            'best_val_dice': best_val_dice, 'lr': get_lr(optimizer),
            'epoch_time_s': ep_time, 'ran_val': run_val, 'is_best': is_best,
            'n_train_cases': len(epoch_cases),
        }
        for c in range(1,6):
            rec[f'val_dice_c{c}'] = val_per.get(c, float('nan'))
        history.append(rec)
        pd.DataFrame(history).to_csv(hist_csv, index=False)

        elapsed_total = time.time() - total_start
        print(f'>>> Ep[{epoch:03d}/{args.epochs:03d}] TrainLoss={train_loss:.4f} ValDice={val_dice:.4f} Best={best_val_dice:.4f} EpTime={ep_time:.0f}s Elapsed={elapsed_total/60:.1f}min', flush=True)

    total_min = (time.time() - total_start) / 60
    print(f'\n=== DONE === {total_min:.1f} min | Best Full-Vol Val Dice: {best_val_dice:.4f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='EXPERIMENT_002 Fixed Training with Case Rotation')
    p.add_argument('--data_dir',           default='data/oaizib')
    p.add_argument('--splits_dir',         default='splits')
    p.add_argument('--checkpoints_dir',    default='experiments/experiment_002/checkpoints')
    p.add_argument('--logs_dir',           default='experiments/experiment_002/logs')
    p.add_argument('--epochs',             type=int,   default=30)
    p.add_argument('--warmup_epochs',      type=int,   default=3)
    p.add_argument('--val_every',          type=int,   default=5)
    p.add_argument('--batch_size',         type=int,   default=2)
    p.add_argument('--lr',                 type=float, default=3e-4)
    p.add_argument('--patch_d',            type=int,   default=48)
    p.add_argument('--patch_h',            type=int,   default=128)
    p.add_argument('--patch_w',            type=int,   default=128)
    p.add_argument('--patches_per_volume', type=int,   default=6)
    p.add_argument('--cases_per_epoch',    type=int,   default=30)
    p.add_argument('--max_val_cases',      type=int,   default=3)
    p.add_argument('--resume',             action='store_true')
    args = p.parse_args()
    train(args)
