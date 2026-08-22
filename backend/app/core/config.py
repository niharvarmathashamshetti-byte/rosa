"""
=============================================================================
ROSA Knee AI — Application Settings
=============================================================================
Centralized configuration for the backend application.

All paths, limits, and feature flags are defined here so they can be
changed in one place without touching business logic.

Uses pydantic-settings to support environment variables:
    export ROSA_DATA_DIR=/custom/path
    export ROSA_MAX_UPLOAD_MB=500
=============================================================================
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings.

    Values can be overridden by environment variables prefixed with ROSA_.
    For example, setting ROSA_DATA_DIR=/my/data in the environment
    will override the default data_dir path.
    """

    # -----------------------------------------------------------------------
    # Project paths
    # -----------------------------------------------------------------------
    # Project root is two levels up from this file (backend/app/core/config.py)
    project_root: Path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir: Path = Path("")  # Set in model_post_init
    cases_dir: Path = Path("")  # Set in model_post_init
    outputs_dir: Path = Path("")  # Set in model_post_init

    # -----------------------------------------------------------------------
    # Upload limits
    # -----------------------------------------------------------------------
    max_upload_size_mb: int = 2000  # Max upload size in megabytes
    allowed_upload_extensions: list[str] = [".zip", ".nii", ".nii.gz", ".nrrd", ".mha"]

    # -----------------------------------------------------------------------
    # Preprocessing defaults
    # -----------------------------------------------------------------------
    default_target_spacing: list[float] = [0.5, 0.5, 0.5]
    default_window_center: int = 400
    default_window_width: int = 1500

    # -----------------------------------------------------------------------
    # Model configuration
    # -----------------------------------------------------------------------
    # Which segmentation model to use: "placeholder" or "nnunet" (future)
    segmentation_model: str = "placeholder"

    # -----------------------------------------------------------------------
    # API configuration
    # -----------------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"
    app_title: str = "ROSA Knee AI"
    app_version: str = "0.1.0"
    debug: bool = True

    # -----------------------------------------------------------------------
    # CORS — allow frontend dev server
    # -----------------------------------------------------------------------
    cors_origins: list[str] = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternative React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    class Config:
        env_prefix = "ROSA_"

    def model_post_init(self, __context) -> None:
        """Compute derived paths after initialization."""
        if self.data_dir == Path(""):
            self.data_dir = self.project_root / "data"
        if self.cases_dir == Path(""):
            self.cases_dir = self.data_dir / "cases"
        if self.outputs_dir == Path(""):
            self.outputs_dir = self.project_root / "outputs"

        # Ensure critical directories exist
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "meshes").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "plans").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Singleton settings instance — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
