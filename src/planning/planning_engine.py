"""
=============================================================================
ROSA Knee AI — Surgical Planning Engine
=============================================================================
Interface for future anatomical measurements and surgical planning.

This module will eventually compute:
    - Femoral Head Center
    - Ankle Center
    - Knee Center
    - Mechanical Axis
    - Hip-Knee-Ankle angle (HKA)
    - Medial Proximal Tibial Angle (MPTA)
    - Posterior Tibial Slope
    - Sweet Spot for implant placement

CURRENT STATUS: Not available.
    All planning calculations require validated segmentation masks and
    anatomical landmarks. Since the segmentation model is not trained yet,
    no measurements can be computed.

IMPORTANT:
    - Do NOT invent measurements
    - Do NOT return fake angles or distances
    - Return clear "not available" status for every metric
=============================================================================
"""

from typing import Any


class PlanningEngine:
    """
    Surgical planning calculations.

    Currently returns "not available" for all measurements.
    Will be implemented after:
        1. Segmentation model is trained
        2. Landmark detection is validated
        3. Measurement algorithms are verified against clinical standards
    """

    def compute_plan(self, segmentation_mask=None, landmarks=None) -> dict[str, Any]:
        """
        Compute all planning measurements.

        Args:
            segmentation_mask: 3D segmentation labels (future input).
            landmarks: Detected anatomical landmarks (future input).

        Returns:
            Dictionary with planning results and status.
        """
        # Check prerequisites
        if segmentation_mask is None:
            return self._unavailable_response(
                "Planning calculations require a validated segmentation mask. "
                "Run segmentation first."
            )

        if landmarks is None:
            return self._unavailable_response(
                "Planning calculations require detected anatomical landmarks. "
                "Landmark detection has not been implemented yet."
            )

        # Future: actual planning computations go here
        return self._unavailable_response(
            "Planning algorithms have not been implemented yet."
        )

    def get_status(self) -> dict[str, Any]:
        """Return planning engine status."""
        return {
            "status": "not_available",
            "message": (
                "Planning calculations require validated segmentation "
                "and landmarks. Neither is available yet."
            ),
            "measurements": {
                "hka": {"status": "not_available", "value": None, "unit": "degrees"},
                "mpta": {"status": "not_available", "value": None, "unit": "degrees"},
                "posterior_slope": {"status": "not_available", "value": None, "unit": "degrees"},
                "sweet_spot": {"status": "not_available", "value": None, "unit": None},
                "mechanical_axis": {"status": "not_available", "value": None, "unit": "mm"},
                "femoral_head_center": {"status": "not_available", "value": None, "unit": "mm"},
                "ankle_center": {"status": "not_available", "value": None, "unit": "mm"},
                "knee_center": {"status": "not_available", "value": None, "unit": "mm"},
            },
        }

    def _unavailable_response(self, message: str) -> dict[str, Any]:
        """Helper to build an 'unavailable' response."""
        result = self.get_status()
        result["message"] = message
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
planning_engine = PlanningEngine()
