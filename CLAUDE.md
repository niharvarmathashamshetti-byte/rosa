# CLAUDE.md — Instructions for Claude Code

This file tells Claude how to behave in this repository, including the
automatic Git/GitHub workflow.

---

## 1. Project Summary

**ROSA Knee AI** — Surgical planning platform for ROSA-assisted Total Knee
Arthroplasty. Backend: FastAPI. Frontend: React + Vite + Three.js. Imaging:
SimpleITK / NumPy. 3D: Marching Cubes → STL meshes.

See `README.md` for the full architecture, status, and quick-start.

---

## 2. Git & GitHub Workflow (Automatic)

After every meaningful development task, Claude MUST:

1. **Review changes** with `git status` and `git diff`.
2. **Run appropriate tests/checks** before committing
   (e.g. `python -m pytest backend/tests/ -v` for backend changes,
   `npm run build` or type-checks for frontend changes).
3. **Stage only the relevant files.** Never include unrelated changes the
   user did not ask for.
4. **Write a clear commit message** describing *what* changed and *why*.
5. **Commit** the staged files.
6. **Push** the commit to the configured GitHub remote.
7. **Never force-push** (`--force`, `--force-with-lease`).
8. **Never rewrite history** (no interactive rebase, no `reset --hard` on
   shared branches).
9. **If a push fails**, diagnose the problem (auth, network, non-fast-forward,
   branch protection, etc.) and report what happened — do not silently retry
   or ignore.

### Hard exclusions — NEVER commit or push

- `.env`, `.env.*` (except `.env.example`)
- API keys, passwords, auth tokens, credentials, private keys
  (`*.pem`, `*.key`, `*.pfx`, `*.p12`, `id_rsa`, `id_ed25519`, etc.)
- Medical / patient data and imaging files:
  `*.nii`, `*.nii.gz`, `*.dcm`, `*.DCM`, `*.dicom`, `*.npz`, `*.nrrd`,
  `*.mhd`, `*.mha`, `*.raw`, anything under `data/`
- Model weights: `*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.hdf5`, `*.ckpt`,
  `*.safetensors`, `*.pb`, `*.tflite`, `weights/`, `checkpoints/`
- Large datasets / binaries: `*.tar`, `*.tar.gz`, `*.tgz`, `*.7z`,
  `*.parquet`, `*.arrow`, `*.feather`

These are already enforced via `.gitignore`. If a new sensitive file type
appears in the project, update `.gitignore` and `CLAUDE.md` together.

### Confirmation policy

- **First push** of this repo: Claude MUST show the user the remote, branch,
  staged file list, and `.gitignore` changes before pushing, and ask for
  explicit confirmation.
- **Subsequent pushes**: after the user has confirmed the workflow once,
  Claude auto-commits and auto-pushes for each meaningful task.
- **Anytime something unexpected shows up in `git status`**
  (untracked files unrelated to the current task, or files that look
  sensitive), Claude MUST stop and ask before staging them.

### Remote

- **Repository:** `git@github.com:niharvarmathashamshetti-byte/rosa.git`
- **Default branch:** `main`
- **Auth:** SSH — the user must have their SSH key registered with GitHub.

---

## 3. Repo Layout (quick reference)

```
backend/    FastAPI app (api/, app/, core/, schemas/, services/, tests/)
frontend/   React + Vite + Three.js UI
src/        Domain modules (io/, preprocessing/, model/, reconstruction/, planning/, ...)
data/       Local-only patient data (gitignored)
outputs/    Generated artifacts (gitignored)
configs/    YAML configs
notebooks/  Jupyter notebooks
scripts/    Utility scripts
```

---

## 4. Running the App

```bash
# Backend
uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm run dev
```

Open http://localhost:5173. API docs: http://localhost:8000/docs.

---

## 5. Testing

```bash
python -m pytest backend/tests/ -v
```

Add a test alongside the change for any non-trivial backend logic.
