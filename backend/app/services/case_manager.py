"""
=============================================================================
ROSA Knee AI — Case Manager Service
=============================================================================
Filesystem-based case storage and management.

Each case is stored as a directory under data/cases/:

    data/cases/
    └── case_000001/
        ├── case.json          ← metadata (status, timestamps, validation)
        ├── raw/               ← uploaded files (DICOM slices, NIfTI, etc.)
        ├── preprocessed/      ← resampled/windowed volumes
        ├── segmentation/      ← segmentation masks
        └── meshes/            ← STL files

Design note:
    All storage logic is in this one class. To swap in a database later,
    replace only this class — the API endpoints don't change.
=============================================================================
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.app.core.config import settings
from backend.app.core.exceptions import CaseNotFoundError
from backend.app.schemas.case import CaseInfo, CaseSummary, CaseValidation


class CaseManager:
    """
    Manages case lifecycle: create, read, update, delete.

    All data is stored on the filesystem. No database required.
    """

    def __init__(self, cases_dir: Optional[Path] = None):
        """
        Args:
            cases_dir: Root directory for case storage.
                       Defaults to settings.cases_dir (data/cases/).
        """
        self.cases_dir = cases_dir or settings.cases_dir
        self.cases_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Case ID generation
    # -------------------------------------------------------------------
    def _generate_case_id(self) -> str:
        """
        Generate a unique sequential case ID like 'case_000001'.

        Scans existing case directories to find the next available number.
        This is simple and readable — fine for a research prototype.
        """
        existing = sorted(self.cases_dir.glob("case_*"))
        if not existing:
            return "case_000001"

        # Find the highest existing number
        max_num = 0
        for d in existing:
            try:
                num = int(d.name.split("_")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue

        return f"case_{max_num + 1:06d}"

    # -------------------------------------------------------------------
    # Case directory structure
    # -------------------------------------------------------------------
    def _case_dir(self, case_id: str) -> Path:
        """Get the root directory for a case."""
        return self.cases_dir / case_id

    def _case_json_path(self, case_id: str) -> Path:
        """Get the path to a case's metadata JSON file."""
        return self._case_dir(case_id) / "case.json"

    def _ensure_case_exists(self, case_id: str) -> Path:
        """Raise CaseNotFoundError if the case directory doesn't exist."""
        case_dir = self._case_dir(case_id)
        if not case_dir.exists():
            raise CaseNotFoundError(case_id)
        return case_dir

    # -------------------------------------------------------------------
    # CRUD operations
    # -------------------------------------------------------------------
    def create_case(self) -> str:
        """
        Create a new case with empty subdirectories.

        Returns:
            The generated case_id (e.g., 'case_000001').
        """
        case_id = self._generate_case_id()
        case_dir = self._case_dir(case_id)

        # Create the case directory structure
        (case_dir / "raw").mkdir(parents=True, exist_ok=True)
        (case_dir / "preprocessed").mkdir(parents=True, exist_ok=True)
        (case_dir / "segmentation").mkdir(parents=True, exist_ok=True)
        (case_dir / "meshes").mkdir(parents=True, exist_ok=True)

        # Write initial metadata
        metadata = {
            "case_id": case_id,
            "status": "uploaded",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_format": "unknown",
            "processing_status": "not_started",
            "model_status": "unavailable",
            "validation": None,
        }
        self._save_metadata(case_id, metadata)

        return case_id

    def get_case(self, case_id: str) -> CaseInfo:
        """
        Load case metadata from case.json and return as CaseInfo.

        Raises:
            CaseNotFoundError: if the case doesn't exist.
        """
        self._ensure_case_exists(case_id)
        metadata = self._load_metadata(case_id)
        case_dir = self._case_dir(case_id)

        # Check what outputs actually exist on disk
        has_preprocessed = any((case_dir / "preprocessed").iterdir())
        has_segmentation = any((case_dir / "segmentation").iterdir())
        has_meshes = any((case_dir / "meshes").glob("*.stl"))

        # Build validation object if present
        validation = None
        if metadata.get("validation"):
            validation = CaseValidation(**metadata["validation"])

        return CaseInfo(
            case_id=metadata["case_id"],
            status=metadata.get("status", "unknown"),
            created_at=metadata.get("created_at", "unknown"),
            input_format=metadata.get("input_format", "unknown"),
            image_size=metadata.get("image_size"),
            voxel_spacing=metadata.get("voxel_spacing"),
            orientation=metadata.get("orientation"),
            physical_size=metadata.get("physical_size"),
            processing_status=metadata.get("processing_status", "not_started"),
            model_status="unavailable",  # Always unavailable until model is trained
            validation=validation,
            has_preprocessed=has_preprocessed,
            has_segmentation=has_segmentation,
            has_meshes=has_meshes,
        )

    def list_cases(self) -> list[CaseSummary]:
        """
        List all cases with summary information.

        Returns:
            List of CaseSummary objects, sorted by case_id.
        """
        summaries = []
        for case_dir in sorted(self.cases_dir.glob("case_*")):
            if not case_dir.is_dir():
                continue
            try:
                metadata = self._load_metadata(case_dir.name)
                summaries.append(CaseSummary(
                    case_id=metadata["case_id"],
                    status=metadata.get("status", "unknown"),
                    created_at=metadata.get("created_at", "unknown"),
                    input_format=metadata.get("input_format", "unknown"),
                    image_size=metadata.get("image_size"),
                    processing_status=metadata.get("processing_status", "not_started"),
                ))
            except Exception:
                # Skip corrupted case directories
                continue

        return summaries

    def delete_case(self, case_id: str) -> None:
        """
        Delete a case and all its data.

        Raises:
            CaseNotFoundError: if the case doesn't exist.
        """
        case_dir = self._ensure_case_exists(case_id)
        shutil.rmtree(case_dir)

    def update_case(self, case_id: str, **updates) -> None:
        """
        Update specific fields in the case metadata.

        Args:
            case_id: The case to update.
            **updates: Key-value pairs to update in case.json.
        """
        self._ensure_case_exists(case_id)
        metadata = self._load_metadata(case_id)
        metadata.update(updates)
        self._save_metadata(case_id, metadata)

    def get_raw_dir(self, case_id: str) -> Path:
        """Get the raw upload directory for a case."""
        self._ensure_case_exists(case_id)
        return self._case_dir(case_id) / "raw"

    def get_preprocessed_dir(self, case_id: str) -> Path:
        """Get the preprocessed output directory for a case."""
        self._ensure_case_exists(case_id)
        return self._case_dir(case_id) / "preprocessed"

    def get_segmentation_dir(self, case_id: str) -> Path:
        """Get the segmentation output directory for a case."""
        self._ensure_case_exists(case_id)
        return self._case_dir(case_id) / "segmentation"

    def get_meshes_dir(self, case_id: str) -> Path:
        """Get the meshes output directory for a case."""
        self._ensure_case_exists(case_id)
        return self._case_dir(case_id) / "meshes"

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _save_metadata(self, case_id: str, metadata: dict) -> None:
        """Write metadata to case.json."""
        json_path = self._case_json_path(case_id)
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self, case_id: str) -> dict:
        """Read metadata from case.json."""
        json_path = self._case_json_path(case_id)
        if not json_path.exists():
            raise CaseNotFoundError(case_id)
        with open(json_path, "r") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Singleton instance — import this in endpoints
# ---------------------------------------------------------------------------
case_manager = CaseManager()
