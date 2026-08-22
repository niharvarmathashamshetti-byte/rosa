# ROSA Knee AI — Surgical Planning Platform

**AI-Based Knee Bone Segmentation, 3D Reconstruction and Surgical Planning for ROSA-Assisted Total Knee Arthroplasty**

---

## 📌 Project Status

- **Module 0 (Data Foundation):** Complete (Inspection, Discovery, QC, Preprocessing tools).
- **Application Architecture:** Complete (FastAPI Backend + React / Vite / Three.js Frontend).
- **AI Segmentation Model:** **NOT TRAINED YET**. The backend is configured with a modular `SegmentationModel` strategy interface (`src/model/inference.py`). It returns clear `model_unavailable` indicators without generating fake predictions.
- **3D Mesh Reconstruction:** Active and functional via Marching Cubes (`src/reconstruction/mesh_generator.py`). Can extract and view 3D STL meshes from real uploaded segmentation masks (`.npz`, `.nii.gz`).
- **Surgical Planning Engine:** Architecture & schemas ready (`src/planning/planning_engine.py`). Returns `not_available` until validated landmarks and segmentation masks are computed.

---

## 🏗️ Architecture

```text
                                FRONTEND (React + Vite + Three.js)
                                                │
                                                ▼ (HTTP / REST)
                                      FASTAPI BACKEND
                                                │
               ┌────────────────────────────────┼────────────────────────────────┐
               ▼                                ▼                                ▼
       Case Management                   CT Slice Viewer                  Processing API
    (data/cases/{id}/raw)            (2D Orthogonal PNGs)                       │
               │                                                 ┌──────────────┼──────────────┐
               ▼                                                 ▼              ▼              ▼
     Quality Control & Validation                         Preprocessing    Segmentation  Reconstruction
          (src/quality_control)                        (0.5mm Isotropic)        │         (Marching Cubes)
                                                                                ▼              │
                                                                           Placeholder         ▼
                                                                           Model (NOW)    Interactive 3D
                                                                                │          STL Meshes
                                                                                ▼          (Three.js)
                                                                           nnU-Net (LATER)
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.10+
- Node.js v18+ & npm

---

### Step 1: Backend Setup & Launch

1. Open a terminal in the project root:
   ```bash
   cd ROSA_Knee_AI
   ```

2. (Optional) Activate your virtual environment:
   ```powershell
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   ```

3. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Launch the FastAPI server:
   ```bash
   uvicorn backend.app.main:app --reload --port 8000
   ```

5. Verify:
   - Health Check: `http://localhost:8000/health`
   - Interactive Swagger API Docs: `http://localhost:8000/docs`

---

### Step 2: Frontend Setup & Launch

1. Open a second terminal:
   ```bash
   cd ROSA_Knee_AI/frontend
   ```

2. Install npm dependencies (if not done yet):
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

4. Open your browser at:
   ```
   http://localhost:5173
   ```

---

## 🧪 Testing the Application

### 1. Automated Backend Tests
Run the pytest suite to verify all endpoints, validation, and error states:
```bash
python -m pytest backend/tests/ -v
```

### 2. Manual End-to-End Workflow in the UI
1. **Open Dashboard** at `http://localhost:5173`.
2. Click **Upload Case** and drag-and-drop a `.npz` file, `.zip` of DICOM slices, or `.nii.gz` file.
3. The system validates dimensions, format, and spatial metadata.
4. Click **Inspect** to navigate to the Case Detail page.
5. In the **2D Slice Viewer**, navigate through Axial, Coronal, and Sagittal cross-sections and adjust windowing (Bone HU / Soft Tissue).
6. Click **Run Preprocessing** to standardize resolution.
7. Click **Run AI Segmentation** — notice the system honestly reports `Model Status: NOT TRAINED`.
8. Click **Run 3D Mesh Reconstruction** — Marching Cubes generates real 3D STL meshes for all labeled structures.
9. Click **View Results & Planning** to inspect the interactive Three.js 3D viewport (drag to rotate, scroll to zoom) and download STL files.

---

## 🔒 Security & Data Privacy

- All patient data stays strictly local inside `data/cases/`.
- De-identified internal case identifiers (`case_000001`) are automatically assigned.
- No analytics or external network calls are performed.
- This is a research prototype, not a certified medical device.

---

## 📁 Repository Structure

```text
ROSA_Knee_AI/
│
├── data/
│   ├── raw/                 # Original reference dataset files
│   ├── cases/               # Active patient cases (raw, preproc, meshes)
│   ├── interim/
│   └── processed/
│
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application entry
│   │   ├── api/v1/          # Endpoints: cases, processing, viewer, model
│   │   ├── core/            # Settings, error handlers, config
│   │   ├── schemas/         # Pydantic data schemas
│   │   └── services/        # CaseManager, CTValidator
│   ├── tests/               # Pytest suite
│   └── requirements.txt     # Backend Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/      # CTViewer, ThreeDViewer, StatusBadge, Navbar
│   │   ├── pages/           # Dashboard, Upload, CaseDetail, Results
│   │   ├── services/api.js  # REST API Client
│   │   ├── App.jsx          # Root view state
│   │   └── App.css          # Medical UI styling
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   ├── io/                  # DICOM, NIfTI, NPZ loaders
│   ├── preprocessing/       # Resampling, Windowing, Cropping
│   ├── visualization/       # Slices, Overlays, Matplotlib plots
│   ├── quality_control/     # Dataset scanning & validation
│   ├── model/               # SegmentationModel & Placeholder
│   ├── reconstruction/      # Marching Cubes STL generator
│   └── planning/            # ROSA planning engine interface
│
├── configs/
│   └── preprocessing.yaml
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_preprocessing_demo.ipynb
│
├── requirements.txt
└── README.md
```
