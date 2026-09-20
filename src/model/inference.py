"""
=============================================================================
ROSA Knee AI — Segmentation Model Interface
=============================================================================
Strategy Pattern Implementation:
    SegmentationModel (abstract)
        ├── PlaceholderSegmentationModel
        └── TrainedKneeSegmentationModel (Trained 3D Residual U-Net)
=============================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path
import numpy as np
import torch

from src.model.network import KneeUNet3D
from src.model.predictor import KneeSegmentor
from src.model.metrics import LABEL_NAMES


class SegmentationModel(ABC):
    @abstractmethod
    def load(self, model_path: Optional[Path] = None) -> None:
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        pass


class PlaceholderSegmentationModel(SegmentationModel):
    def __init__(self):
        self._loaded = False
        self.model_name = "PlaceholderSegmentationModel"

    def load(self, model_path: Optional[Path] = None) -> None:
        self._loaded = False

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        return {
            "status": "model_unavailable",
            "mask": None,
            "labels": None,
            "message": "Placeholder model: train and load a real model first.",
        }

    def unload(self) -> None:
        self._loaded = False

    def is_available(self) -> bool:
        return False

    def get_status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": "unavailable",
            "trained": False,
            "message": "Segmentation model is not loaded.",
        }


class TrainedKneeSegmentationModel(SegmentationModel):
    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.model_name = "TrainedKneeUNet3D"
        self._loaded = False
        self.segmentor = None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else Path("checkpoints/best_model.pt")

        if self.checkpoint_path.exists():
            self.load(self.checkpoint_path)

    def load(self, model_path: Optional[Path] = None) -> None:
        path = Path(model_path) if model_path else self.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = KneeUNet3D(in_channels=1, num_classes=6, channels=(16, 32, 64, 128)).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        self.segmentor = KneeSegmentor(model, device=self.device, patch_size=(32, 64, 64), overlap=0.2)
        self._loaded = True
        self.checkpoint_path = path

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        if not self._loaded or self.segmentor is None:
            return {
                "status": "error",
                "mask": None,
                "labels": None,
                "message": "Model is not loaded."
            }

        pred_mask = self.segmentor.predict_volume(image.astype(np.float32))
        return {
            "status": "success",
            "mask": pred_mask,
            "labels": LABEL_NAMES,
            "message": "Segmentation completed successfully."
        }

    def unload(self) -> None:
        self.segmentor = None
        self._loaded = False

    def is_available(self) -> bool:
        return self._loaded

    def get_status(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": "available" if self._loaded else "unavailable",
            "trained": True,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
            "device": str(self.device)
        }


def get_segmentation_model(model_type: str = "trained") -> SegmentationModel:
    if model_type == "trained":
        ckpt_p = Path("checkpoints/best_model.pt")
        if ckpt_p.exists():
            return TrainedKneeSegmentationModel(checkpoint_path=ckpt_p)
        return PlaceholderSegmentationModel()
    elif model_type == "placeholder":
        return PlaceholderSegmentationModel()
    else:
        return PlaceholderSegmentationModel()
