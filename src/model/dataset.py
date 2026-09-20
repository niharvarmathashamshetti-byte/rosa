import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

class KneeMRIDataset3D(Dataset):
    """
    High-Performance 3D Knee MRI Dataset with in-memory caching and balanced patch extraction.
    """
    def __init__(
        self,
        case_ids: list,
        data_dir: str = 'data/oaizib',
        patch_size: tuple = (32, 64, 64),
        patches_per_volume: int = 4,
        fg_prob: float = 0.85,
        augment: bool = True,
        is_train: bool = True,
        preload: bool = True
    ):
        self.case_ids = case_ids
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'imagesTr'
        self.lbl_dir = self.data_dir / 'labelsTr'
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.fg_prob = fg_prob
        self.augment = augment
        self.is_train = is_train

        self.samples = []
        for cid in self.case_ids:
            img_p = self.img_dir / f'{cid}_0000.nii.gz'
            lbl_p = self.lbl_dir / f'{cid}.nii.gz'
            if img_p.exists() and lbl_p.exists():
                self.samples.append((str(img_p), str(lbl_p), cid))

        # In-memory cache
        self.cache = {}
        if preload and len(self.samples) <= 100:
            print(f'Preloading {len(self.samples)} volumes into RAM for fast patch extraction...', flush=True)
            for idx, (img_p, lbl_p, cid) in enumerate(self.samples):
                img_nii = nib.load(img_p)
                lbl_nii = nib.load(lbl_p)
                img = img_nii.get_fdata().astype(np.float32)
                lbl = lbl_nii.get_fdata().astype(np.uint8)
                img = self._normalize_image(img)
                
                # Precompute foreground indices for ultra-fast sampling
                fg_indices = np.argwhere(lbl > 0)
                self.cache[idx] = (img, lbl, fg_indices)
            print(f'Successfully cached {len(self.cache)} volumes.', flush=True)

    def __len__(self):
        if self.is_train:
            return len(self.samples) * self.patches_per_volume
        return len(self.samples)

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, (0.5, 99.5))
        img = np.clip(img, p1, p99)
        mean = np.mean(img)
        std = np.std(img) + 1e-6
        img = (img - mean) / std
        return img.astype(np.float32)

    def _extract_patch(self, img: np.ndarray, lbl: np.ndarray, fg_indices: np.ndarray):
        D, H, W = img.shape
        pD, pH, pW = self.patch_size
        
        pD = min(pD, D)
        pH = min(pH, H)
        pW = min(pW, W)

        if self.fg_prob > 0 and len(fg_indices) > 0 and np.random.rand() < self.fg_prob:
            center = fg_indices[np.random.choice(len(fg_indices))]
            d_start = int(np.clip(center[0] - pD // 2, 0, D - pD))
            h_start = int(np.clip(center[1] - pH // 2, 0, H - pH))
            w_start = int(np.clip(center[2] - pW // 2, 0, W - pW))
        else:
            d_start = np.random.randint(0, max(1, D - pD + 1))
            h_start = np.random.randint(0, max(1, H - pH + 1))
            w_start = np.random.randint(0, max(1, W - pW + 1))

        img_patch = img[d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        lbl_patch = lbl[d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        return img_patch, lbl_patch

    def _augment(self, img: np.ndarray, lbl: np.ndarray):
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=0).copy()
            lbl = np.flip(lbl, axis=0).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=1).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
            lbl = np.flip(lbl, axis=2).copy()

        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            img = img * scale + shift

        return img, lbl

    def __getitem__(self, idx):
        sample_idx = idx // self.patches_per_volume if self.is_train else idx
        
        if sample_idx in self.cache:
            img, lbl, fg_indices = self.cache[sample_idx]
            cid = self.samples[sample_idx][2]
        else:
            img_path, lbl_path, cid = self.samples[sample_idx]
            img = nib.load(img_path).get_fdata().astype(np.float32)
            lbl = nib.load(lbl_path).get_fdata().astype(np.uint8)
            img = self._normalize_image(img)
            fg_indices = np.argwhere(lbl > 0)

        if self.is_train:
            img_patch, lbl_patch = self._extract_patch(img, lbl, fg_indices)
            if self.augment:
                img_patch, lbl_patch = self._augment(img_patch, lbl_patch)
            
            img_tensor = torch.from_numpy(img_patch).unsqueeze(0).float()
            lbl_tensor = torch.from_numpy(lbl_patch).long()
            return img_tensor, lbl_tensor, cid
        else:
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            lbl_tensor = torch.from_numpy(lbl).long()
            return img_tensor, lbl_tensor, cid
