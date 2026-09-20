import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

class KneeSegmentor:
    def __init__(self, model, device='cpu', patch_size=(48, 96, 96), overlap=0.5):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap
        self.model.to(self.device)
        self.model.eval()

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, (0.5, 99.5))
        img = np.clip(img, p1, p99)
        mean = np.mean(img)
        std = np.std(img) + 1e-6
        img = (img - mean) / std
        return img.astype(np.float32)

    @torch.no_grad()
    def predict_volume(self, image_np: np.ndarray) -> np.ndarray:
        norm_img = self._normalize_image(image_np)
        input_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Sliding window inference
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self.patch_size,
            sw_batch_size=1,
            predictor=self.model,
            overlap=self.overlap,
            mode='gaussian'
        )
        
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return pred_mask

    def predict_nifti(self, nifti_path: str, output_path: str = None) -> tuple:
        nii = nib.load(str(nifti_path))
        data = nii.get_fdata().astype(np.float32)
        pred_mask = self.predict_volume(data)
        
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            pred_nii = nib.Nifti1Image(pred_mask, affine=nii.affine, header=nii.header)
            nib.save(pred_nii, str(out_p))
            
        return pred_mask, nii.affine, nii.header
