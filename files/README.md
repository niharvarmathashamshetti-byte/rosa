# Knee OAI-ZIB Segmentation Pipeline (nnU-Net v2)

Module 1 of the ROSA-assisted TKA project: automatic 3D knee MRI
segmentation (femur, femoral cartilage, tibia, medial/lateral tibial
cartilage). Research pipeline only — no clinical claims, no ROSA
robot integration in this module.

## Current data status

`config.py` currently points at:
- `IMAGES_TR = C:\Users\91950\Downloads\imagesTr`
- `LABELS_TR = C:\Users\91950\Downloads\labelsTr`
- `IMAGES_TS = None`  (not provided yet)
- `LABELS_TS = None`  (not provided yet)

Everything up through cross-validation training/evaluation/visualization/
3D reconstruction can run with just the training set. The official
held-out **test** evaluation (spec Step 17) is blocked until you add the
two test paths to `config.py` and re-run
`02_convert_to_nnunet_format.py`.

## Setup (once)

```bash
pip install -r requirements.txt --break-system-packages
```
Install the correct CUDA build of PyTorch for your GPU *before* this,
from https://pytorch.org/get-started/locally/, if `pip install nnunetv2`
doesn't already pull one that matches your driver.

## Run order

```bash
# 1. Confirm GPU / CUDA / nnU-Net are actually usable on this machine
python 00_environment_check.py

# 2. Full dataset integrity verification (19 checks, spec Section 5).
#    STOPS on any critical issue — does not silently continue.
python 01_dataset_verification.py

# 3. Build the nnU-Net raw dataset + dataset.json (copies files, does
#    not touch your original Downloads folders)
python 02_convert_to_nnunet_format.py

# 4. nnU-Net's own integrity check + planning + preprocessing
python 03_run_nnunet_pipeline.py --step verify

# 5. Optional short smoke test before committing to full training
python 03_run_nnunet_pipeline.py --step smoketest --fold 0

# 6. Train (repeat for folds 0-4 for full 5-fold CV; at minimum run fold 0)
python 03_run_nnunet_pipeline.py --step train --fold 0
# or, for all folds sequentially:
python 03_run_nnunet_pipeline.py --step train_all_folds

# 7. Evaluate on cross-validation (this is what you use to judge/select
#    the model — NEVER the test set)
python 04_evaluate_and_visualize.py --mode validation

# 8. Only after you add IMAGES_TS/LABELS_TS to config.py and re-run
#    step 3, and only once you've picked your final model from step 7:
python 02_convert_to_nnunet_format.py   # re-run to copy test images in
python 04_evaluate_and_visualize.py --mode test

# 9. 3D reconstruction demo (marching cubes -> STL) for a case or all cases
python 05_reconstruct_3d.py --source validation --all

# 10. Assemble the final report from whatever has actually been produced
python 06_generate_final_report.py
```

## Design notes / guardrails baked into this code

- **No step trains or evaluates on data that hasn't passed integrity
  checks.** `02_convert_to_nnunet_format.py` refuses to run without a
  saved, passing `dataset_integrity_report.json` from step 1.
- **Test-set leakage guard:** `imagesTs`/`labelsTs` are never referenced
  during planning, preprocessing, or training — only in
  `04_evaluate_and_visualize.py --mode test`, which also refuses to
  run a second time without `--force`, since spec Section 12/14
  requires the official test set to be touched exactly once.
- **No fabricated metrics.** Every number in `final_report.txt` is read
  back from a JSON/CSV file a previous script actually wrote; missing
  sections say `NOT AVAILABLE — <reason>` instead of a placeholder
  number.
- **Label mapping is fixed** to the 6-class map you specified
  (`config.py::LABEL_MAP`) and the pipeline hard-stops if any label
  file contains a value outside `{0,...,5}` rather than silently
  remapping it.
- **GPU/VRAM is detected, never assumed.** If your GPU can't fit the
  default `3d_fullres` plan, `00_environment_check.py` warns you but
  does not change your configuration — you decide whether to fall back
  to `3d_lowres` (edit `NNUNET_CONFIGURATIONS_TO_TRY` in `config.py`).

## What still needs YOUR input

1. Confirm `PROJECT_ROOT` in `config.py` is where you want ~tens of GB
   of preprocessed data + checkpoints written (nnU-Net preprocessing +
   results can be large).
2. Add `IMAGES_TS` / `LABELS_TS` paths once you have them.
3. Decide how many folds you can afford to train given your GPU/time
   budget (`FOLDS` in `config.py`); nnU-Net's default is 5.
