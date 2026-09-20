import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.network import KneeUNet3D
from src.model.dataset import KneeMRIDataset3D
from src.model.losses import DiceCELoss

def fast_tensor_dice(preds: torch.Tensor, targets: torch.Tensor, num_classes: int = 6):
    fg_dices = []
    fg_ious = []
    class_dices = {}
    for c in range(1, num_classes):
        p = (preds == c)
        g = (targets == c)
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        p_sum = p.sum().item()
        g_sum = g.sum().item()
        d = (2.0 * inter) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0 if (p_sum == 0 and g_sum == 0) else 0.0
        iou = inter / union if union > 0 else 1.0 if (p_sum == 0 and g_sum == 0) else 0.0
        class_dices[c] = float(d)
        fg_dices.append(d)
        fg_ious.append(iou)
    return float(np.mean(fg_dices)), float(np.mean(fg_ious)), class_dices

def train(args):
    torch.set_num_threads(8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== Starting Ultra-Fast ROSA Knee 3D Model Training ===', flush=True)
    print(f'Device: {device} | CPU threads: {torch.get_num_threads()}', flush=True)

    splits_dir = Path(args.splits_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    logs_dir = Path(args.logs_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_dir / 'train.txt') as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open(splits_dir / 'validation.txt') as f:
        val_ids = [line.strip() for line in f if line.strip()]

    if args.max_train_cases > 0:
        train_ids = train_ids[:args.max_train_cases]
    if args.max_val_cases > 0:
        val_ids = val_ids[:args.max_val_cases]

    print(f'Training dataset: {len(train_ids)} cases | Validation dataset: {len(val_ids)} cases', flush=True)

    train_dataset = KneeMRIDataset3D(
        case_ids=train_ids,
        data_dir=args.data_dir,
        patch_size=(args.patch_d, args.patch_h, args.patch_w),
        patches_per_volume=args.patches_per_volume,
        fg_prob=0.85,
        augment=True,
        is_train=True,
        preload=True
    )

    val_dataset = KneeMRIDataset3D(
        case_ids=val_ids,
        data_dir=args.data_dir,
        patch_size=(args.patch_d, args.patch_h, args.patch_w),
        patches_per_volume=4,
        fg_prob=0.85,
        augment=False,
        is_train=True,
        preload=True
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = KneeUNet3D(
        in_channels=1,
        num_classes=6,
        channels=(16, 32, 64, 128),
        dropout=0.1
    ).to(device)

    class_weights = torch.tensor([0.2, 1.0, 2.5, 1.0, 3.0, 3.0], dtype=torch.float32).to(device)
    criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5, class_weights=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_dice = -1.0
    history = []

    last_ckpt_path = checkpoints_dir / 'last_model.pt'
    best_ckpt_path = checkpoints_dir / 'best_model.pt'

    if args.resume and last_ckpt_path.exists():
        print(f'Resuming training from {last_ckpt_path}...', flush=True)
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_dice = ckpt.get('best_val_dice', -1.0)
        print(f'Resumed at epoch {start_epoch}, previous best val dice: {best_val_dice:.4f}', flush=True)

    total_start = time.time()
    total_steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for step, (images, targets, _) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            if step % 10 == 0 or step == total_steps_per_epoch:
                print(f'Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:02d}/{total_steps_per_epoch:02d}] Batch Loss: {loss.item():.4f}', flush=True)

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        current_lr = scheduler.get_last_lr()[0]

        # Fast Validation on Val Patches
        model.eval()
        val_losses = []
        val_dices = []
        val_ious = []
        femur_dices = []
        tibia_dices = []
        cart_dices = []

        with torch.no_grad():
            for v_imgs, v_tgts, _ in val_loader:
                v_imgs = v_imgs.to(device)
                v_tgts = v_tgts.to(device)

                v_logits = model(v_imgs)
                v_loss = criterion(v_logits, v_tgts)
                val_losses.append(v_loss.item())

                v_preds = torch.argmax(v_logits, dim=1)
                m_dice, m_iou, c_dices = fast_tensor_dice(v_preds, v_tgts)

                val_dices.append(m_dice)
                val_ious.append(m_iou)
                femur_dices.append(c_dices[1])
                cart_dices.append(c_dices[2])
                tibia_dices.append(c_dices[3])

        val_loss = float(np.mean(val_losses))
        val_mean_dice = float(np.mean(val_dices))
        val_mean_iou = float(np.mean(val_ious))
        femur_dice = float(np.mean(femur_dices))
        tibia_dice = float(np.mean(tibia_dices))
        cart_dice = float(np.mean(cart_dices))
        epoch_time = time.time() - epoch_start

        is_best = val_mean_dice > best_val_dice
        if is_best:
            best_val_dice = val_mean_dice

        ckpt_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mean_dice': val_mean_dice,
            'best_val_dice': best_val_dice,
            'config': vars(args)
        }

        torch.save(ckpt_state, last_ckpt_path)
        if is_best:
            torch.save(ckpt_state, best_ckpt_path)
            print(f'*** [BEST MODEL CHECKPOINT SAVED] Epoch {epoch:02d} Val Mean FG Dice: {val_mean_dice:.4f} ***', flush=True)

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mean_dice': val_mean_dice,
            'val_mean_iou': val_mean_iou,
            'femur_dice': femur_dice,
            'tibia_dice': tibia_dice,
            'fem_cart_dice': cart_dice,
            'lr': current_lr,
            'epoch_time_s': epoch_time,
            'is_best': is_best
        }
        history.append(record)
        pd.DataFrame(history).to_csv(logs_dir / 'training_history.csv', index=False)

        print(f'>>> Epoch [{epoch:02d}/{args.epochs:02d}] Summary: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | '
              f'Val FG Dice={val_mean_dice:.4f} | Femur={femur_dice:.3f} | Tibia={tibia_dice:.3f} | Cart={cart_dice:.3f} | '
              f'Time={epoch_time:.1f}s\n', flush=True)

    print(f'=== Training Completed Successfully ===\nTotal time: {(time.time() - total_start)/60:.2f} mins. Peak Val FG Dice: {best_val_dice:.4f}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/oaizib')
    parser.add_argument('--splits_dir', type=str, default='splits')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patch_d', type=int, default=32)
    parser.add_argument('--patch_h', type=int, default=64)
    parser.add_argument('--patch_w', type=int, default=64)
    parser.add_argument('--patches_per_volume', type=int, default=3)
    parser.add_argument('--max_train_cases', type=int, default=45)
    parser.add_argument('--max_val_cases', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    train(args)
