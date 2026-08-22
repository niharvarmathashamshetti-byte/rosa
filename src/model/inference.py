"""
=============================================================================
ROSA Knee AI — Segmentation Model Interface
=============================================================================
This module defines the abstract interface that ALL segmentation models must
implement. This is the Strategy Pattern:

    SegmentationModel (abstract)
        ├── PlaceholderSegmentationModel  ← NOW (returns "not available")
        └── NNUNetSegmentationModel       ← LATER (after training)

The backend service only knows about the abstract base class.
When we train nnU-Net, we create a new implementation and swap it in
WITHOUT changing the API endpoints, frontend, or case management.

IMPORTANT:
    - Do NOT generate fake predictions
    - Do NOT return made-up confidence scores
    - The placeholder honestly reports that no model is available
=============================================================================
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path

import numpy as np


class SegmentationModel(ABC):
    """
    Abstract base class for all segmentation models.

    Any segmentation model (nnU-Net, Attention U-Net, etc.) must implement
    these three methods. This ensures we can swap models without changing
    the rest of the codebase.
    """

    @abstractmethod
    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load model weights from disk into memory (and GPU if available).

        Args:
            model_path: Path to the model checkpoint/weights directory.
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Run segmentation inference on a 3D volume.

        Args:
            image: 3D numpy array (preprocessed CT volume).

        Returns:
            Dictionary containing:
                - 'status': 'success' or 'unavailable' or 'error'
                - 'mask': 3D numpy array of segmentation labels (or None)
                - 'labels': dict mapping label IDs to names (or None)
                - 'message': human-readable status message
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release model from memory (and GPU)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        pass

    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """Return model status information."""
        pass


class PlaceholderSegmentationModel(SegmentationModel):
    """
    Placeholder model that honestly reports it is not available.

    This is the ONLY model implementation until we train nnU-Net.
    It does NOT:
        - Generate fake segmentation masks
        - Return made-up confidence scores
        - Pretend to detect anatomical structures

    It DOES:
        - Report that the model is not trained
        - Return clear status messages
        - Allow the rest of the pipeline to work around model absence
    """

    def __init__(self):
        self._loaded = False
        self.model_name = "PlaceholderSegmentationModel"

    def load(self, model_path: Optional[Path] = None) -> None:
        """Placeholder: no model to load."""
        print(f"[{self.model_name}] No trained model available to load.")
        self._loaded = False

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """
        Placeholder: returns 'unavailable' status.

        Does NOT return fake predictions.
        """
        return {
            "status": "model_unavailable",
            "mask": None,
            "labels": None,
            "message": (
                "Segmentation model has not been trained yet. "
                "No predictions can be generated at this time. "
                "Train the nnU-Net model first, then replace this "
                "placeholder with NNUNetSegmentationModel."
            ),
        }

    def unload(self) -> None:
        """Placeholder: nothing to unload."""
        self._loaded = False

    def is_available(self) -> bool:
        """Always returns False — no trained model exists."""
        return False

    def get_status(self) -> dict[str, Any]:
        """Return clear status about model unavailability."""
        return {
            "model_name": self.model_name,
            "status": "unavailable",
            "trained": False,
            "message": "Segmentation model has not been trained yet.",
        }


# ---------------------------------------------------------------------------
# Future: NNUNetSegmentationModel
# ---------------------------------------------------------------------------
# class NNUNetSegmentationModel(SegmentationModel):
#     """
#     nnU-Net inference model.
#
#     This will be implemented after training is complete.
#     It will:
#         - Load trained nnU-Net weights
#         - Run inference on preprocessed CT volumes
#         - Return segmentation masks with label mapping
#
#     The API, frontend, and case management will NOT need to change.
#     """
#     pass


# ---------------------------------------------------------------------------
# Factory function — returns the correct model based on config
# ---------------------------------------------------------------------------
def get_segmentation_model(model_type: str = "placeholder") -> SegmentationModel:
    """
    Factory that returns the appropriate segmentation model.

    Args:
        model_type: "placeholder" (now) or "nnunet" (future).

    Returns:
        An instance of SegmentationModel.
    """
    if model_type == "placeholder":
        return PlaceholderSegmentationModel()
    # elif model_type == "nnunet":
    #     return NNUNetSegmentationModel()
    else:
        print(f"Unknown model type '{model_type}', using placeholder.")
        return PlaceholderSegmentationModel()
