"""
=============================================================================
ROSA Knee AI — Comprehensive Backend Tests
=============================================================================
Unit and integration tests for FastAPI endpoints:
  - Health check
  - Model status
  - Case upload (valid & invalid)
  - Case metadata & retrieval
  - Preprocessing & segmentation unavailable status
  - 3D mesh reconstruction from mask
  - Results query
=============================================================================
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.case_manager import case_manager

client = TestClient(app)


def test_health_check():
    """Verify GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_status_unavailable():
    """Verify that model status honestly reports 'unavailable' and 'not trained'."""
    response = client.get("/api/v1/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["trained"] is False
    assert "not been trained" in data["message"]


def test_invalid_upload_rejected():
    """Verify that uploading an unsupported file format (.txt) is rejected with 400."""
    response = client.post(
        "/api/v1/cases",
        files={"file": ("notes.txt", b"patient notes", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["message"]


def test_case_lifecycle_with_segmentation_mask():
    """
    Full integration test:
      1. Create synthetic 3D mask
      2. Upload to backend
      3. Verify validation
      4. Run preprocessing
      5. Verify segmentation returns 'model_unavailable'
      6. Run 3D reconstruction -> generates real STL meshes
      7. Query results
      8. Download STL
      9. Delete case
    """
    # 1. Create synthetic 3D mask
    mask = np.zeros((30, 30, 20), dtype=np.uint8)
    mask[5:25, 5:25, 5:15] = 1  # Femur placeholder mask
    mask[10:20, 10:20, 8:12] = 2 # Tibia placeholder mask

    buf = io.BytesIO()
    np.savez_compressed(buf, x=mask)
    buf.seek(0)

    # 2. Upload case
    upload_res = client.post(
        "/api/v1/cases",
        files={"file": ("test_study.npz", buf.getvalue(), "application/octet-stream")}
    )
    assert upload_res.status_code == 201
    case_info = upload_res.json()
    case_id = case_info["case_id"]
    assert case_info["input_format"] == "NPZ"
    assert case_info["image_size"] == [30, 30, 20]

    # 3. Get case details
    get_res = client.get(f"/api/v1/cases/{case_id}")
    assert get_res.status_code == 200

    # 4. Run preprocessing
    preproc_res = client.post(f"/api/v1/cases/{case_id}/preprocess")
    assert preproc_res.status_code == 200

    # 5. Run segmentation (Must report model_unavailable)
    seg_res = client.post(f"/api/v1/cases/{case_id}/segment")
    assert seg_res.status_code == 200
    assert seg_res.json()["segmentation_status"] == "model_unavailable"

    # 6. Run reconstruction
    recon_res = client.post(f"/api/v1/cases/{case_id}/reconstruct")
    assert recon_res.status_code == 200
    assert recon_res.json()["num_meshes"] == 2

    # 7. Get results
    results_res = client.get(f"/api/v1/cases/{case_id}/results")
    assert results_res.status_code == 200
    results = results_res.json()
    assert results["segmentation"]["status"] == "model_unavailable"
    assert results["reconstruction"]["status"] == "available"
    assert len(results["reconstruction"]["meshes"]) == 2
    assert results["planning"]["status"] == "not_available"

    # 8. Download STL mesh
    mesh_filename = results["reconstruction"]["meshes"][0]["filename"]
    mesh_res = client.get(f"/api/v1/cases/{case_id}/mesh/{mesh_filename}")
    assert mesh_res.status_code == 200
    assert len(mesh_res.content) > 0

    # 9. Clean up / Delete
    del_res = client.delete(f"/api/v1/cases/{case_id}")
    assert del_res.status_code == 200

    # Confirm 404 after deletion
    assert client.get(f"/api/v1/cases/{case_id}").status_code == 404


def test_run_preprocessing_script_creates_preprocessed_nifti(tmp_path):
    """The preprocessing CLI should resample an input NIfTI file to the target spacing."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    image = sitk.GetImageFromArray(np.ones((10, 20, 30), dtype=np.float32))
    image.SetSpacing((2.0, 1.0, 1.0))
    sitk.WriteImage(image, str(data_dir / "sample_volume.nii.gz"))

    output_dir = tmp_path / "preprocessed"
    script_path = project_root / "scripts" / "run_preprocessing.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    out_file = output_dir / "sample_volume_preprocessed.nii.gz"
    assert out_file.exists(), result.stdout

    out_image = sitk.ReadImage(str(out_file))
    assert tuple(round(v, 3) for v in out_image.GetSpacing()) == (0.5, 0.5, 0.5)
