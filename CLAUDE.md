# CLAUDE.md — ROSA Knee AI

> Persistent project context for Claude Code. Read this file at the start of every session to understand the project.

---

## 1. Project Goal

**ROSA Knee AI** is a medical-imaging / AI pipeline that generates **patient-specific 3D knee models** from medical imaging to support **ROSA robotic knee surgery / Total Knee Arthroplasty (TKA)**.

It is currently a **research / development prototype** — not a certified medical device.

The three primary anatomical targets are:

- **Femur**
- **Tibia**
- **Patella**

---

## 2. Dataset

- **Source dataset:** OAI (Osteoarthritis Initiative) **Tissue Segmentations** dataset — knee MRI data with tissue/anatomical segmentations.
- The dataset is **NOT** in this repository.
- All medical imaging files must be stored under `data/` (gitignored).
- **Do not commit, push, or upload any patient / private medical data.**
- The only on-disk example case is `data/cases/case_000001/` (synthetic test mask, 30×30×20 uint8 with 3 labels, plus generated STLs).

---

## 3. High-Level Pipeline

```text
Medical imaging (DICOM / NIfTI / NPZ)
        ↓
Upload → CT validation
        ↓
Preprocessing (resample, windowing, optional cropping)
        ↓
3D segmentation        ← NOT TRAINED YET (placeholder)
        ↓
Post-processing
        ↓
3D mesh reconstruction (Marching Cubes → STL)
        ↓
Landmark detection     ← PLANNED
        ↓
Knee alignment / IKA calculations
        ↓
Patient-specific surgical plan (JSON)
        ↓
ROSA-compatible output ← NOT VERIFIED YET
```

**Important:** Do not claim that any output is "officially ROSA-compatible" unless the implementation has actually been verified against the ROSA specification. The current state is a research scaffold only.

---

## 4. Current Project Status (Reality Check)

| Component | Status |
|---|---|
| Dataset inspection / QC tools (`src/quality_control`, `src/io`) | **Implemented** |
| Preprocessing library (`src/preprocessing/ct_preprocessing.py`) | **Skeleton functions** — `apply_window`, `resample_image`, `crop_volume`, `normalize_intensity`. Functions print "to be used AFTER data verification" and have not been wired to real data verification. |
| CT validation + upload via FastAPI (`backend/`) | **Implemented and tested** |
| Case management (filesystem-based, `CaseManager`) | **Implemented** |
| 2D slice viewer endpoint (PNG via Matplotlib, all 3 planes + windowing) | **Implemented** |
| Preprocessing API endpoint (`/preprocess`) | **Implemented** — resamples NIfTI/DICOM to 0.5 mm; copies NPZ through (no spacing). |
| Segmentation model (`src/model/inference.py`) | **Placeholder only.** `PlaceholderSegmentationModel` honestly returns `model_unavailable`. No real model. |
| 3D mesh reconstruction (`src/reconstruction/mesh_generator.py`) | **Implemented and working** — Marching Cubes → binary STL per label. Verified on synthetic mask. |
| Interactive 3D viewer (Three.js, `frontend/src/components/ThreeDViewer`) | **Implemented** — loads generated STLs, rotate/zoom, download. |
| Surgical planning engine (`src/planning/planning_engine.py`) | **Skeleton only.** Returns `not_available` for every measurement. |
| Landmark detection (PCA, sphere fitting, FHC, ankle center) | **PLANNED / NOT IMPLEMENTED** |
| Alignment / IKA (HKA, MPTA, posterior slope, sweet spot) | **PLANNED / NOT IMPLEMENTED** |
| nnU-Net / 3D Attention U-Net training | **PLANNED / NOT IMPLEMENTED** |
| ROSA compatibility verification | **PLANNED / NOT IMPLEMENTED** |
| Tests (`backend/tests/test_api.py`) | **Implemented** — end-to-end lifecycle test against TestClient. |

### What works right now
- Backend server boots, `/health` returns 200, model status reports "unavailable".
- Uploading a `.npz` mask or `.nii.gz` / `.zip(DICOM)` / `.nrrd` / `.mha` via the UI creates a `case_NNNNNN` directory.
- `CTValidator` extracts spatial metadata from NIfTI/DICOM; NPZ has no spatial metadata.
- The 2D slice viewer renders PNG slices with adjustable bone/soft-tissue windowing.
- `/preprocess` resamples NIfTI/DICOM to **0.5 mm isotropic** and saves `volume_preprocessed.nii.gz`.
- `/segment` returns a clear `model_unavailable` response.
- `/reconstruct` runs Marching Cubes on the uploaded NPZ mask and produces one STL per non-zero label.
- The 3D viewer renders the generated STLs interactively.
- The full upload → preprocess → segment → reconstruct → download flow is covered by `backend/tests/test_api.py`.

### What does not work yet
- A real segmentation model — there is no trained `nnunet` / `3D Attention U-Net`.
- Landmark detection, alignment calculations, surgical plan JSON.
- NPZ cannot be properly resampled because NPZ has no spacing/origin metadata.

---

## 5. Technology Stack (currently used)

### Backend (`backend/`)
- **Python 3.x** (CPython — `__pycache__` shows 3.13).
- **FastAPI** for the HTTP API (uploads, processing, viewer, model status).
- **Uvicorn** ASGI server (`uvicorn[standard]`).
- **Pydantic v2** + `pydantic-settings` for schemas and config.
- **SimpleITK** for medical image I/O (DICOM, NIfTI, NRRD, MHA).
- **NumPy** for arrays.
- **Matplotlib** (Agg backend) for server-rendered slice PNGs.
- **scikit-image** (`skimage.measure.marching_cubes`) for 3D mesh extraction.
- **PyYAML** for config.
- **python-multipart**, **aiofiles** for file uploads.
- **pytest**, **httpx**, **pytest-asyncio** for tests.

### Domain library (`src/`)
- Uses **SimpleITK**, **NumPy**, **scikit-image**, **matplotlib**, **pandas**, **tqdm**, **PyYAML**, **pydicom**, **nibabel**.
- **Torch, MONAI, nnU-Net, VTK, trimesh, open3d** are explicitly **NOT installed yet** (see comment in `requirements.txt`).

### Frontend (`frontend/`)
- **React 19** + **Vite 6**.
- **Three.js 0.174** for 3D rendering (`STLLoader`).
- **lucide-react** for icons.
- Vite dev server proxies `/api` and `/health` → `http://127.0.0.1:8000`.

---

## 6. Repo Layout

```text
ROSA_Knee_AI/
├── backend/                 FastAPI app
│   ├── app/
│   │   ├── main.py          FastAPI entry — lifespan, CORS, /health, mounts v1 router
│   │   ├── api/v1/
│   │   │   ├── router.py           v1 aggregator + GET /model/status
│   │   │   ├── cases.py            POST/GET/DELETE /cases
│   │   │   ├── processing.py       /preprocess, /segment, /reconstruct, /results, /mesh/{name}
│   │   │   └── viewer.py           /slices/{axis}/{index} (PNG), /volume-info
│   │   ├── core/
│   │   │   ├── config.py           pydantic-settings (env prefix ROSA_)
│   │   │   └── exceptions.py       RosaBaseException + handlers
│   │   ├── schemas/
│   │   │   ├── case.py             CaseCreate / CaseInfo / CaseSummary / CaseValidation
│   │   │   └── responses.py        HealthResponse, ErrorResponse, ModelStatusResponse
│   │   └── services/
│   │       ├── case_manager.py     Filesystem case CRUD
│   │       └── ct_validator.py     DICOM / NIfTI / NPZ validation
│   ├── tests/test_api.py           End-to-end lifecycle test
│   └── requirements.txt
│
├── frontend/                React + Vite + Three.js
│   ├── src/
│   │   ├── main.jsx, App.jsx, App.css
│   │   ├── pages/           Dashboard, Upload, CaseDetail, Results
│   │   ├── components/
│   │   │   ├── Layout/      Navbar, StatusBadge
│   │   │   ├── CTViewer/    2D orthogonal slice viewer (windowing UI)
│   │   │   └── ThreeDViewer/ Interactive Three.js STL viewer
│   │   └── services/api.js  REST client for all endpoints
│   ├── vite.config.js       dev server + /api proxy
│   └── package.json
│
├── src/                     Domain library (used by both backend and scripts)
│   ├── io/                  npz_loader, dicom_loader, nifti_loader
│   ├── preprocessing/       ct_preprocessing (apply_window, resample_image, crop_volume, normalize_intensity)
│   ├── visualization/       volume_visualizer (slice / overlay / histogram / 3D preview)
│   ├── quality_control/     dataset_checker (classify, scan, report)
│   ├── model/               inference.py — SegmentationModel ABC + PlaceholderSegmentationModel
│   ├── reconstruction/      mesh_generator — mask_to_mesh, generate_all_meshes, binary STL writer
│   └── planning/            planning_engine — returns "not_available" for all measurements
│
├── configs/preprocessing.yaml   target spacing 0.5 mm, bone window 400/1500, soft tissue 40/400
├── scripts/
│   ├── inspect_dataset.py        scans data dir, generates CSV report + summary
│   └── run_preprocessing.py      CLI wrapper — currently only prints warning, does not preprocess
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_preprocessing_demo.ipynb
│
├── data/                    gitignored — patient data
│   ├── raw/, interim/, processed/, cases/
├── outputs/                 gitignored — figures, reports, processed, meshes, plans
├── requirements.txt         top-level Python deps (no torch/monai yet)
├── .gitignore               comprehensive — covers medical files, weights, secrets
└── CLAUDE.md                this file
```

### On-disk case layout (per `data/cases/case_NNNNNN/`)
```
case_NNNNNN/
├── case.json          # metadata (status, validation, processing_status)
├── raw/               # uploaded files
├── preprocessed/      # resampled volumes (NIfTI or NPZ pass-through)
├── segmentation/      # model output masks (empty — no model yet)
└── meshes/            # generated .stl files
```

---

## 7. Key Design Decisions (Already in the Code)

1. **Strategy pattern for segmentation model** — `SegmentationModel` ABC with `PlaceholderSegmentationModel` (now) and a future `NNUNetSegmentationModel`. The API endpoints and frontend never need to change when swapping models.
2. **No fake predictions** — every code path that would otherwise produce invented values (segmentation, planning) instead returns a clear "unavailable" / "not_available" status with a human-readable message. This is enforced by `test_api.py`.
3. **Filesystem-based case storage** — no database; cases live under `data/cases/` as directories with a `case.json` metadata file. `CaseManager` is the single abstraction so a DB can replace it later.
4. **CT validation is read-only** — `CTValidator` uses SimpleITK to read metadata without modifying anything.
5. **2D slices rendered server-side** — `/slices/{axis}/{index}` returns PNG bytes (Matplotlib Agg backend) so the client doesn't need a GPU.
6. **Three.js for 3D** — `STLLoader` reads binaries from the backend `/mesh/{name}` endpoint.
7. **Configuration via `pydantic-settings`** with env-var prefix `ROSA_` (e.g. `ROSA_DATA_DIR=/custom`).
8. **CORS locked to dev ports** — `http://localhost:5173`, `http://127.0.0.1:5173`, plus `3000` variants.

---

## 8. Configuration Values Currently Used

- **Target resampling spacing:** `(0.5, 0.5, 0.5)` mm isotropic — set in `configs/preprocessing.yaml` and hard-coded in `backend/app/api/v1/processing.py` (`target_spacing = (0.5, 0.5, 0.5)`).
- **Bone window:** center 400 HU, width 1500 HU (covers roughly −350 to +1150 HU).
- **Soft-tissue window:** center 40 HU, width 400 HU.
- **Max upload size:** 2000 MB (configurable via `ROSA_MAX_UPLOAD_MB`).
- **Marching Cubes `step_size`:** 2 in the API (faster preview), 1 by default in the library.

If you change these values, update both `configs/preprocessing.yaml` and the hard-coded fallbacks in `processing.py` and `viewer.py` together.

---

## 9. Running the App

```bash
# Backend
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev

# Tests
python -m pytest backend/tests/ -v

# Dataset inspection
python scripts/inspect_dataset.py --data-dir data/raw
```

- Backend: `http://localhost:8000` — docs at `/docs`, health at `/health`.
- Frontend: `http://localhost:5173` (proxies `/api` and `/health` to backend).

---

## 10. Important Commands

| Task | Command |
|---|---|
| Run backend | `uvicorn backend.app.main:app --reload --port 8000` |
| Run frontend | `cd frontend && npm run dev` |
| Run all tests | `python -m pytest backend/tests/ -v` |
| Run a single test | `python -m pytest backend/tests/test_api.py::test_health_check -v` |
| Inspect a dataset | `python scripts/inspect_dataset.py --data-dir data/raw` |
| Build frontend | `cd frontend && npm run build` |
| Curl model status | `curl http://localhost:8000/api/v1/model/status` |

---

## 11. Git & GitHub Workflow (Automatic)

After every meaningful development task, Claude MUST:

1. Review changes with `git status` and `git diff`.
2. Run appropriate tests/checks before committing (pytest for backend; `npm run build` for frontend).
3. Stage **only the relevant files** — never include unrelated changes.
4. Write a clear commit message describing *what* and *why*.
5. Commit the staged files.
6. Push the commit to the configured GitHub remote.
7. Never force-push (`--force`, `--force-with-lease`).
8. Never rewrite history (no interactive rebase, no `reset --hard` on shared branches).
9. If a push fails, diagnose (auth, network, non-fast-forward, branch protection) and report — do not silently retry.

### Remote
- **Repository:** `git@github.com:niharvarmathashamshetti-byte/rosa.git`
- **Default branch:** `main`
- **Auth:** SSH

### Hard exclusions — NEVER commit or push
- `.env`, `.env.*` (except `.env.example`)
- API keys, passwords, auth tokens, credentials, private keys (`*.pem`, `*.key`, `*.pfx`, `*.p12`, `id_rsa`, `id_ed25519`)
- Medical / patient imaging: `*.nii`, `*.nii.gz`, `*.dcm`, `*.DCM`, `*.dicom`, `*.npz`, `*.nrrd`, `*.mhd`, `*.mha`, `*.raw`, anything under `data/`
- Model weights: `*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.hdf5`, `*.ckpt`, `*.safetensors`, `*.pb`, `*.tflite`, `weights/`, `checkpoints/`
- Large datasets / binaries: `*.tar`, `*.tar.gz`, `*.tgz`, `*.7z`, `*.parquet`, `*.arrow`, `*.feather`

These are already enforced via `.gitignore` — keep it that way. If a new sensitive file type appears, update both `.gitignore` and this file together.

### Confirmation policy
- **First push** of this repo: Claude MUST show the user the remote, branch, staged file list, and `.gitignore` changes before pushing, and ask for explicit confirmation.
- **Subsequent pushes**: auto-commit + auto-push for each meaningful task.
- **Anytime something unexpected shows up in `git status`** (untracked files unrelated to the current task, or files that look sensitive), Claude MUST stop and ask before staging them.

---

## 12. Development Rules

Before modifying code:
- **Understand the existing implementation first.** Read the relevant files end-to-end before changing them.
- **Do not unnecessarily rewrite working code.** The current code intentionally avoids fake predictions and keeps the model as a placeholder.
- **Prefer simple, readable implementations** — the user is still learning. Comments are valued.
- **Explain important changes** before making major architectural changes.
- **Keep functions modular** — one concern per function where practical.
- **Avoid unnecessary dependencies.**
- **Preserve existing working functionality.** If unsure, add a new module/path instead of overwriting.

In particular:
- **Do not remove the "no fake predictions" guarantee** in `src/model/inference.py` or `src/planning/planning_engine.py` — they are required by `test_api.py` and by the project's honesty contract.
- **Do not assume what label IDs mean** in segmentation masks (the code uses `label_N` filenames and never maps 1→femur, 2→tibia, 3→patella without verification). The OAI label mapping has not been verified in this repository.

---

## 13. Known Issues / Inconsistencies Found

1. **Inconsistent label naming across UI copy and code.** `Results.jsx` and `case.json` mention "Femur, Tibia, Patella, Cartilage" as expected outputs, but the code has no actual mapping from NPZ label IDs to anatomical names — the project has **not** verified the OAI label legend. The `label_1.stl`, `label_2.stl`, `label_3.stl` files in `data/cases/case_000001/meshes/` are placeholders.
2. **`run_preprocessing.py` is a stub.** It parses the config and prints a warning, but does not run preprocessing. All preprocessing is actually triggered through the FastAPI endpoint instead.
3. **`VolumeVisualizer` (`src/visualization`) is not used by the backend or frontend.** The 2D viewer uses a Matplotlib-based server endpoint (`backend/app/api/v1/viewer.py`) instead. `volume_visualizer.py` remains a standalone library for notebooks/scripts.
4. **`windowing` config has hard-coded duplicates.** The bone window values (400/1500) are present in `configs/preprocessing.yaml`, `backend/app/core/config.py`, `backend/app/api/v1/viewer.py`, and `frontend/src/components/CTViewer/CTViewer.jsx`. Changing one without the others will desync UI and backend.
5. **`src/preprocessing/ct_preprocessing.py` functions are "skeleton only".** They work for basic operations (e.g. `resample_image` is called by the API), but the docstrings warn they are not yet validated on the real dataset.
6. **NIfTI/NPZ data should be in `(Z, Y, X)` numpy order** for the SimpleITK pipeline; the API handles this with `sitk.GetArrayFromImage`, but be careful with direct numpy slicing elsewhere.
7. **`StudyDescription`, `Manufacturer`, etc. in `dicom_loader.py` are read but not surfaced** in the API. The frontend does not display DICOM clinical tags.
8. **3D viewer uses left-click + drag to rotate** but does not use an OrbitControls instance — interaction is custom and may feel less polished than a full Three.js controls implementation.
9. **No data leakage check between train/val/test** — the project has no training loop yet, so this is not blocking, but it will matter once the segmentation model is trained.
10. **No CI** — tests are run manually via `pytest`/`npm run build`.

If you find a new one, document it here.

---

## 14. Planned / Not Yet Implemented

- **Real segmentation model** — train 3D Attention U-Net / nnU-Net on the OAI Tissue Segmentations dataset. Add `NNUNetSegmentationModel` in `src/model/inference.py` and swap via `get_segmentation_model("nnunet")` (the factory and config already support this).
- **Verifying label meanings** in the OAI dataset and exposing them in `label_names` in the reconstruction path.
- **Auto knee localization / cropping** — `cropping.method: "auto"` in `preprocessing.yaml` is referenced but the implementation does not exist.
- **Landmark detection** — PCA, sphere fitting, femoral head center, ankle center.
- **Alignment / IKA calculations** — HKA, MPTA, posterior slope, sweet spot, mechanical axis.
- **Patient-specific surgical plan JSON** export.
- **ROSA compatibility verification** — until verified, never claim "ROSA-compatible" outputs.
- **Watertight STL generation** — current Marching Cubes output is not guaranteed watertight. Would need `trimesh` or similar.
- **Auth, user accounts, multi-user case isolation** — currently single-user, all cases in one `data/cases/` directory.
- **DICOM clinical tag display** in the frontend.
- **GPU model server** (the current placeholder runs on CPU; nnU-Net will need GPU selection and model warm-up).
- **CI/CD** — automated tests on push.

---

## 15. Quick Pointers for Future Claude Sessions

- The segmentation model lives in `src/model/inference.py`. To replace the placeholder, implement the `SegmentationModel` interface and wire it through `get_segmentation_model(model_type="nnunet")` in `backend/app/api/v1/processing.py`. **Do not** change the response shape.
- The planning engine lives in `src/planning/planning_engine.py`. Replace `_unavailable_response` calls with real computations once landmarks and masks are available. **Do not** return invented angles/distances.
- The reconstruction step expects either a real segmentation mask in `data/cases/{id}/segmentation/*.npz` or an NPZ in the raw upload. The mask is loaded with `key = "x" if "x" in data else list(data.keys())[0]` — preserve that convention.
- The frontend already displays `NOT TRAINED` for the model and `NOT AVAILABLE` for planning values; this is correct, not a bug.
- The 2D viewer windowing UI and the backend windowing endpoint both default to bone window (400/1500). Keep them in sync.

---

## 16. One-Line Summary

> ROSA Knee AI is a research scaffold that already supports upload, validation, resampling, Marching Cubes mesh reconstruction, and an interactive 3D viewer for real masks — but does **not yet** have a trained segmentation model, landmark detection, or alignment calculations. The project is **not** a medical device and its outputs have **not** been verified as ROSA-compatible.
