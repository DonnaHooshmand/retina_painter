# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

Do not commit anything; the user makes all commits. Preserve existing uncommitted
work. The working branch at the 2026-10-21 handoff was `review-reference-ui`;
inspect the current checkout before making changes. The proposal describes the
research direction, but justified improvements may depart from it. Distinguish
implemented features, planned work, and experimentally demonstrated results.

## Local clinical data and curated dataset location

The user requires clinical images, overlays, patient identifiers, and raw clinical
records to remain local and out of assistant context, tool output, and cloud
services. Process real data with local scripts that return aggregate counts only;
do not view real scans through assistant image or screenshot tools. Use synthetic
images for tests. Do not delete or rewrite source projects while preparing data.

The user's annotation practice datasets follow the layout of
`D:\RootPainterSync\curated_datasets\Jon_611_Raphael_612`: clean `images/` plus
matching doctor reference overlays and ground-truth masks. The earlier practice
bundle used the 300 completed reviews in `Raphael_091726` and was created by
`prepare_practice_review_dataset.py`. The current merged version and separate
prepared project are described below. Prior answers must never be preloaded
into a fresh project's editable annotations, decisions, or models.
Preserve verified patient assignments and uncertainty, holding unmapped scans
out of the fresh project. This is distinct from the earlier 613-image frozen
replay export produced by `prepare_curated_training_dataset.py`, which was not
the fresh annotation dataset the user requested. A new folder does not make
previously used development patients an untouched final test cohort.

The current merged reference version is
`D:\RootPainterSync\curated_datasets\Raphael_Updated_20261021_v1`: 316 unique
scans from the original 300 reviews and 69 frozen follow-up reviews. Keep this
version unchanged. `prepare_patient_practice_project.py` creates a separate
blank project using its clean images and the earlier verified patient-assignment
manifest. The current allocation is 255 training scans and 59 validation scans,
with two unresolved identities held outside navigation. It preserves uncertain
reviews, uses explicit train/val filename lists, and starts with a documented
positive/negative warm-up. No prior masks, decisions, or model weights enter
the editable session. Use an absolute local dataset path for this prepared
project: the current client does not normalize `..` before navigation lookup.

## Current handoff and next work (verified 2026-10-21)

Use these existing local artifacts; do not rebuild them without a concrete need:

- Reference dataset:
  `D:\RootPainterSync\curated_datasets\Raphael_Updated_20261021_v1`
- Clean scans: its `images/` folder.
- Human reference pictures: its `ground_truth_audit_overlays/` folder, opened
  separately beside RetinaPainter. Red is explicit clinician foreground; cyan
  is retained model foreground from the completed review. This checkout has no
  built-in prior-label/reference panel.
- Reviewed masks: `ground_truth_binary/` (0/255) and `ground_truth/` (RGBA).
  `doctor_foreground_masks/` contains sparse red marks only. Untouched pixels
  in those sparse masks are unknown.
- Current references and provenance: `manifest.csv`, `ground_truth_status.csv`,
  `review_history.csv`, and `provenance/`. The reference dataset is immutable;
  changed references require a new version.
- Prepared editable project:
  `D:\RootPainterSync\projects\Raphael_Updated_20261021_practice_v1\Raphael_Updated_20261021_practice_v1.seg_proj`
- Sync directory: `D:\RootPainterSync`.

The reference dataset contains 316 scans: 14 present, 286 absent, 16 uncertain.
`prepare_updated_review_dataset.py` merged the original 300 reviews with 69
frozen follow-up reviews by image SHA-256. The follow-up supersedes the earlier
reference for 53 repeated scans and adds 16 image contents. Both readings are
preserved. There are 300 definitive reviewed masks; uncertain scans retain
reference pictures without a definitive mask or binary diagnosis. Reconstructed
references may include retained model predictions and are not independently
adjudicated exhaustive contours.

The original source project is `D:\RootPainterSync\projects\Raphael_091726`.
The frozen follow-up is
`D:\RootPainterSync\projects\Raphael_091726_update\review_exports\20261021_frozen_v1`.
Keep both unchanged. The older 613-image export's manifest at
`D:\RootPainterSync\datasets\curated\_datasets\RIPL_Development_20260925_v1\manifest.csv`
was used only as the verified patient-assignment source for the new project;
it is not the current practice image collection.

Open the prepared `.seg_proj` using **Project > Open project**, not New Project.
`prepare_patient_practice_project.py` preserved the verified assignments:

| Partition | Scans | Patients | Present | Absent | Uncertain |
|---|---:|---:|---:|---:|---:|
| Train | 255 | 112 | 12 | 232 | 11 |
| Validation | 59 | 28 | 2 | 53 | 4 |
| Identity hold | 2 | Unresolved | 0 | 1 | 1 |

Train and validation share zero patients. Identity holds are excluded from
project navigation but remain in the reference dataset. Project-local
`manifest.csv`, `patient_assignments.csv`, `held_images.csv`, `README.txt`, and
`preparation_metadata.json` record the allocation and initial state. Reference
decisions in these preparation tables are not loaded as new completed reviews.

The prepared project uses fresh U-Net weights, training seed 0, and automatic
training after the first ten ordered reviews are completed and the user advances.
Its initial ten are deliberately label-informed: two positive and four negative
training scans, plus two positive and two negative validation scans. Remaining
scans retain source rank. Do not describe this order as random evaluation.
The trainer must be running, and both annotation partitions need painted pixels
for optimization to proceed. An Absent decision alone adds no background pixels.
An Uncertain decision does not exclude already painted pixels from the loss;
leave ambiguous regions unpainted and retain uncertainty in review metadata.

At the last check on 2026-10-21, the prepared project had zero completed reviews,
zero annotation PNGs, and zero model files. Recheck aggregate local state before
acting: the user may have annotated since this handoff. Never reset an existing
session to reproduce that initial state.

Immediate sequence:
1. Verify project paths, trainer availability, and current session state locally.
2. Help the user annotate the first ten scans and confirm automatic training,
   actual epochs, and updated predictions after advancing.
3. Continue another 10–20 scans to check red/green corrections, saving, stable
   patient routing, and missed lesions or excessive false-positive regions.
   Improvement is a question to measure, not a promised pilot outcome.
4. Fix concrete workflow blockers before completing the larger session. Export
   completed reviews to preserve decisions and reviewed snapshots.
5. Implement/verify the clinical evaluation pipeline below before reporting
   performance. Plan an untouched patient-separated final cohort for the paper.
6. Compare architectures in separate fresh projects with consistent patient
   assignments, annotation budgets, and evaluation rules. Do not replace the
   current baseline with another architecture midway through training.

Recent synthetic preparation checks passed: 21 cases in
`painter/tests/test_updated_review_dataset.py` and 16 cases in
`painter/tests/test_patient_practice_project.py`, including annotation routing
and client/navigation/trainer path conversion. These verify workflow integrity,
not clinical performance; no completed clinical benchmark is established here.

### Friday positive-candidate session (2026-10-23)

Prepared on 2026-10-21 for Raphael to annotate additional spreadsheet-positive
scans, distinct from the 316-scan reference dataset and its practice project:

- Dataset: `D:\RootPainterSync\curated_datasets\Raphael_Positive_20261023_v1`
- Clean images: its `images/` folder.
- Open project:
  `D:\RootPainterSync\projects\Raphael_Positive_20261023_v1\Raphael_Positive_20261023_v1.seg_proj`
- Builder: `prepare_positive_annotation_dataset.py`; synthetic tests:
  `painter/tests/test_positive_annotation_dataset.py`.

The user requested new spreadsheet-positive scans, spread across patients.
After discussing 100 versus 200, the prepared project has 200 scans across 121
patients; the first 100 are the priority batch and cover 100 distinct patients.
Every selected scan has image-level RIPL=1 in the local workbook, but this is
candidate selection, not a confirmed clinician diagnosis. Raphael must still
be able to select Absent or Uncertain. No ground-truth masks or reference
overlays exist for these new annotations yet; the ground_truth directory holds
an explanatory README only. Do not manufacture masks from spreadsheet labels.

Selection excluded all prior completed reviews and existing annotation files
found in local projects, including conservative exclusion of empty annotation
placeholders, by filename/stem and available source-image SHA-256. All such
historical source images were available during preparation. Of 522 unique
spreadsheet-positive scans, 187 were excluded for prior review/annotation and
three for patient-identity mismatch, leaving 332 eligible scans across 121
patients. No source projects were modified.

The new project fixes 163 training scans (97 patients) and 37 validation scans
(24 patients), with zero patient overlap. Existing verified patient roles are
preserved for 194 scans; six scans receive seeded roles for newly encountered
patients. The first ten have eight training and two validation patients. It uses
fresh U-Net weights, seed 0, empty annotation/review/model directories, and
automatic training after those ten reviews are completed and the user advances.
Both partitions still need painted pixels and a running trainer. Prior labels
in manifest.csv are audit metadata, not completed review decisions.

This positive-only session adds development annotations; it cannot measure
specificity or population prevalence and is not an untouched final test cohort.
Export completed reviews at the end, then curate a new reference version from
that export. Recheck live project state before acting and never reset it.

### Pre-Friday workflow verification (2026-10-21)

Completed in an isolated synthetic sync directory using the actual Qt painter
and local CUDA U-Net trainer. Ten completed reviews triggered one automatic
training instruction; the normal 612-crop epoch at patch size 572 completed,
validation promoted a checkpoint, and fresh predictions changed and contained
foreground. Saving, closing/reopening, prediction/status display, and export of
12 reviews passed; every exported mask matched its saved review snapshots.
This verifies the workflow, not clinical accuracy. Aggregate file fingerprints
confirmed the Friday dataset and project were unchanged.

Fixes made during this check:
- Windows source-mode trainer discovery now uses `trainer/env/Scripts/python.exe`
  and runs from `trainer/src`. The local painter venv was repaired to use the
  installed Python 3.11.9; no packages were downloaded.
- Automatic save/navigation/close/export of an unchanged completed review now
  preserves its original snapshots, timestamp, and reconstruction mode. Changing
  corrections or decisions, or unchecking/rechecking completion, deliberately
  snapshots the current prediction. A training start alone must not replace a
  previously completed annotation-only reference.
- Status files are processed in modification-time order, with a watermark so a
  locked old message cannot overwrite a newer displayed status. The first GUI
  integration check exposed this bug; verification passed after the fix using
  the same trained synthetic outputs.

The focused painter suite passed 41 tests, including the new
`test_review_gui_saving.py` and `test_server_manager_launch.py`. The opt-in
`trainer/tests/smoke_annotation_session.py` orchestrates the real trainer and
`painter/tests/smoke_annotation_session.py` Qt workflow. It requires a new,
explicitly synthetic sync directory; never pass a clinical sync directory.
Restart the application from this updated source checkout to use these fixes;
existing packaged executables were not rebuilt.

## Project Overview

**RetinaPainter** is a fork of [RootPainter](https://github.com/Abe404/root_painter) adapted for retinal OCT biomarker detection. It uses a client-server architecture where the **painter** (PyQt5 GUI client) and **trainer** (PyTorch server) communicate via JSON instruction files in a shared filesystem directory (the "sync directory"). No network protocol is used—communication works over local filesystem, sshfs, Dropbox, or Google Drive.

RetinaPainter retains U-Net trained from scratch and adds **RETFound ViT-Large**
encoder backends with plain and RFA segmentation decoders. The current prepared
pilot uses U-Net. Foundation-model pretraining is a candidate route to reducing
annotation needs; improved RIPL performance or a specific required sample size
has not been demonstrated by the preparation work. The practical goal is RIPL
localization/segmentation from sparse corrections, with B-scan detection as the
primary clinical endpoint and contour agreement as a secondary diagnostic.

## Architecture

**Two independent Python applications:**

- **`painter/`** — PyQt5 desktop GUI. Users annotate images with brush strokes, view model predictions as overlays, and manage projects/datasets. Entry point: `painter/src/main/python/main.py`. Main window class: `retina_painter.py`. It retains RootPainter's corrective workflow but adds model selection and reproducible trial controls.
- **`trainer/`** — PyTorch training server. Watches the sync directory for instructions, trains models, performs segmentation. Entry point: `trainer/src/main.py`. Core loop: `trainer.py` (`Trainer.main_loop()`).

**Filesystem-based IPC:** The client writes JSON instruction files to `<syncdir>/instructions/`. The trainer polls for these, processes them (train, segment, etc.), and writes segmentation results back to the project directory. The `instructions.py` module in the painter handles creating these files.

**Workstation mode:** `server_manager.py` in the painter can auto-launch a bundled trainer executable, or in dev mode, launch the trainer from `trainer/env/bin/python`.

**Annotation routing — train vs. validation:** New projects pre-assign every seeded filename to a fixed 5:1 train/validation split before annotation begins (`project_order.py`): by patient when the New Project dialog's Patient separation is on (`patient_grouped_train_val_split`), otherwise per file (`fixed_train_val_split`). The `.seg_proj` stores `train_file_names` and `val_file_names`; blank scans or model-dependent corrections cannot shift later filenames between splits. Projects created before this change omit those lists and retain the inherited count-based router. Existing pre-populated annotations are always overwritten in place, preserving externally prepared splits.

**Reproducible trial seed and manifest order:** The New Project dialog has a **Trial seed** (default `0`). For ordinary new projects this single value fixes navigation order, filename-level train/validation membership, random model/decoder initialization, DataLoader ordering, worker RNGs, and Python/NumPy/PyTorch sampling. Training filenames are sorted before random selection so filesystem order cannot change seeded trials. When the selected dataset or its parent contains a `manifest.csv` with `rank` and `filename`, manifest rank overrides the navigation shuffle while the trial seed continues to control model initialization and training randomness. This makes every smaller nested dataset an exact navigation prefix of every larger one. Exact bitwise equality is only expected on the same software/hardware stack.

**Clinician review records:** The painter records an explicit B-scan decision
(`present`, `absent`, or `uncertain`) separately from a `review_complete`
flag. Completed reviews snapshot both the displayed prediction and sparse
correction PNG under `<project>/reviews/`, with SHA-256 hashes in
`review_records.json`. Before training first starts, the random prediction is
ignored for export and the binary mask contains only red clinician foreground.
After training starts, export uses `prediction - green + red`; an explicit
`absent` decision always overrides residual prediction with an empty mask.
Pre-training `present` completion requires red foreground, and `absent`
completion rejects contradictory red foreground. New projects automatically
start training after the first ten ordered scans are all completed and the
clinician advances. Forward navigation requires the current review to be
complete, and the warm-up gate identifies the first missing initial review;
Previous remains available for correction. The manual start action remains
available.
**Project > Export completed reviews** writes the masks into a timestamped
`<project>/review_exports/` directory and writes a review CSV plus consistency
metadata. This makes a correction-free confirmed negative distinguishable
from an unreviewed scan and prevents later live segmentations from changing
completed exports. Each completed review also gets a labeled visual copy under
`reviews/decision_previews/` for human audit; the live UI, source OCT, and
sparse training annotation stay free of label text. See
`docs/clinician_review_workflow.md`.

**Patient separation:** The New Project dialog has a **Patient separation** group, so the fixed split can be made by patient instead of by file. By default the patient is read from one part of the file name: the dialog shows an example name split at `_`, `-`, space and `.` (each can be switched off), pre-selects the first part that groups images (`suggest_patient_part`), and previews the image, patient and per-side counts before anything is created. A CSV with `filename` and `patient_id` columns is the alternative when names do not carry the patient. "No patient grouping" keeps the per-file split and is the starting choice when no part of the names groups images. Settings that cannot give every image a patient, or that put every image in one patient, block project creation.

`patient_grouped_train_val_split` walks the navigation order and sends each newly met patient to validation whenever validation holds less than its 1-in-6 share of the images seen so far. No patient is on both sides; about one image in six is validation throughout a session (the first image still is); a nested-prefix project gives shared patients the same side; and with one image per patient the result equals `fixed_train_val_split`. The `.seg_proj` records `split_method: patient_grouped_…`, `patient_rule`, and `patient_groups` (one opaque number per entry of `file_names`, not the identifiers). The split is fixed before any label exists, so it cannot balance positives between the sides. On the combined development images the first name part reproduces the curated patient list exactly. Projects created before this change, and per-file projects, behave as before.

**Absent scans as clean scans (opt-in):** The New Project dialog has a checkbox, **Use scans marked RIPL absent as clean scans**, off by default and stored as `absent_scans_as_background` in the `.seg_proj` (projects without the key behave as before). With it on, completing a review as "RIPL absent" makes that scan's annotation file background everywhere, on the scan's fixed side, whether or not anything was painted. The trainer needs no change because it only reads annotation files; this is the supervision `--supervision sparse_absent` prepares offline. The strokes the user painted stay in the review snapshot (`reviews/annotation_snapshots/`), and the painter shows those, not a green wash, when the scan is reopened (`RetinaPainter.strokes_path_for_display`). Saving an unchanged absent scan again rewrites nothing, because the trainer treats a changed file time as changed annotations and resets its early-stopping count. Changing the decision, unticking Review complete, or painting red (which blocks an absent review as before) puts the user's strokes back in the annotation file, or removes the file when there are none. "Present" scans keep sparse strokes and "uncertain" scans are not trained on. In the session replay this supervision removed the blob-on-every-scan failure but lost lesion scans (`docs/offline_experiments.md`), so it is an option, not the default. Checks: `painter/tests/smoke_absent_background.py`.

To impose a split prepared outside the painter instead:

- **Prefer explicit project lists:** prepare `train_file_names` and
  `val_file_names` from verified patient assignments, with disjoint membership
  covering every active filename. The current painter honors these lists even
  when annotation folders are empty; blank placeholder PNGs are unnecessary.
- **Legacy compatibility only:** pre-populated annotation PNGs are overwritten
  in place by `get_annot_path` and `maybe_save_annotation`. The current prepared
  project uses explicit lists and empty folders, so do not add placeholders.
  Moving annotations after a model has trained does not undo patient exposure;
  establish assignments before training instead of repairing a split afterwards.
- **Planned: explicit train/val source folders** — see the `train-val-split`
  branch. This feature is absent from the current checkout. A proposed New
  Project checkbox would let the user select separate image directories and
  route annotations by source folder. Patient separation would still need to
  be verified when preparing those directories.

Patient separation must be verified across training, validation, and any final
test cohort, including historical development exposure. A separate test folder
alone does not guarantee independence. The current prepared project has internal
validation only; no untouched final test cohort has been established.

**Interactive checkpoint control:** Before validation contains foreground,
same-project UI predictions use explicitly provisional live weights only while
their background validation loss remains within `0.01` of the retained
checkpoint. A candidate outside that false-positive safety margin continues
training in memory while the UI uses the durable provisional checkpoint.
Background-only CE may provisionally promote checkpoints. Once validation is
foreground-informed, the UI uses the durable best checkpoint, and the
candidate keeps training from where it is, as in RootPainter. Annotation
updates reset early stopping; candidate and incumbent are then re-evaluated on
the same current validation set.

Automatic candidate rollback is off by default (since 2026-11-13). On
annotations that are not changing it fired on ordinary validation noise and
kept discarding training (at least 26 times in one three-hour run).
`--candidate-rollback-patience N` turns it on for a controlled comparison: a
candidate that is worse for N consecutive epochs is then restored from the
best checkpoint, only after at least two validation annotation files contain
foreground, never on background-only validation, and with the count restarted
whenever annotations change.

## Clinical Evaluation Contract

RetinaPainter's primary RIPL outcome is **B-scan-level detection**, not contour
agreement. A held-out B-scan is positive when the clinician reference standard says
that at least one RIPL is present. Final evaluation must report the raw
TP/FP/FN/TN confusion matrix plus sensitivity, specificity, PPV, NPV, balanced
accuracy, and accuracy. Pixel Dice/IoU are secondary engineering diagnostics
only and must not be the headline basis for choosing an architecture.

Do not confuse the clinical endpoint with the training surrogate. The trainer
uses masked combined Dice + 0.3 cross-entropy for optimization, checkpoint
promotion, and early stopping. Hard masked pixel F1 remains diagnostic only;
it cannot block checkpoint updates while a tiny lesion is still below the 0.5
threshold. Background-only validation uses the CE term and emits a warning.
Sparse corrective annotations are not exhaustive and cannot calculate honest
B-scan false negatives or true negatives.

If individual lesions are evaluated later, use one-to-one lesion matching and
report lesion TP/FP/FN, sensitivity, precision, and false positives per B-scan;
lesion-level TN is undefined.

**Implementation gap checked 2026-10-21:** no completed clinical B-scan evaluator
or lesion overlap/distance-tolerance matcher was found in the repository.
`trainer/src/metrics.py` reports supervised-pixel metrics.
`painter/src/main/python/plot_seg_metrics.py` compares model predictions with
their sparsely corrected versions; this is correction-agreement analysis,
not an independent clinical benchmark. Verify or implement the missing pipeline
and synthetic tests before interpreting these outputs as clinical outcomes.

One negative scan with two false predicted regions contributes one B-scan FP;
it may contribute two lesion FPs in a separately defined lesion analysis. A
correct scan-level positive decision does not establish correct localization.
Acceptable localization should use a justified, explicit overlap/proximity rule
appropriate to sparse reference marks and image resolution; no clinically
validated pixel-distance or overlap tolerance has been established here.
Select probability thresholds, component filters, and localization rules using
development/validation data and freeze them before final test evaluation.

Preserve uncertain scans and report them separately from the binary confusion
matrix unless a justified adjudication protocol resolves their labels. Do not
turn uncertain or untouched regions into negatives. Report denominators and
uncertainty: the current validation set contains only two positive scans, so
sensitivity estimates are unstable. Document sample selection and development
history; do not treat this selected collection's positive fraction as population
prevalence or call its clinician-reviewed reconstructions independently drawn
gold-standard contours.

## Models

### U-Net (original, `--model-type unet`, default)

Defined in `unet.py` (`UNetGNRes`). Uses Group Normalization (not Batch Norm) with residual connections. Default input patch size 572×572, output 500×500 (valid convolutions crop 36px per side). Valid patch sizes: 572, 556, 540, ..., 28.

**Training sampler:** `TrainDataset` retains RootPainter's inherited minimum of 612 randomly generated crops per epoch. It builds pools from the sparse corrections already collected. By default, the foreground tile target adapts to `foreground_pool / (foreground_pool + background_pool)`, which gives each pool member equal expected sampling frequency without assuming a balanced image collection. This fraction is a crop-routing probability, not an estimate of image- or patient-level RIPL prevalence; a file containing both red and green corrections is present in both pools. The trainer reports `foreground tile target XX% (adaptive)`, where `XX` is expected to change as annotations accumulate. For example, 7 foreground-bearing and 25 background-bearing pool entries produce a 21.875% target, displayed as 22%; the operational check is `(adaptive)`, not a fixed expected percentage. A fixed fraction remains available for controlled ablations, and sampling falls back automatically when either pool is empty. Each crop must contain its selected correction type in the model's actual output region; for U-Net this also prevents an annotation in the discarded 36-pixel context border from admitting a zero-supervision crop. Pool filenames are sorted and all random choices remain governed by the trial seed.

### RETFound plain decoder (`--model-type retfound`)

Defined across two files:

- **`retfound_vit.py`** — `RETFoundViT`: ViT-Large (patch_size=16, embed_dim=1024, depth=24, num_heads=16) with sin-cos positional embeddings. `forward_features(x)` returns `(B, 196, 1024)` patch tokens (cls token dropped). Also exposes `forward_multi_features(x, indices)` which captures intermediate block outputs for RFA skip connections. Weights match the RETFound checkpoint format exactly.
- **`retfound_model.py`** — `RETFoundSeg`: encoder (`RETFoundViT`) + `_SegDecoder` (4-stage ConvTranspose2d upsampler: 14→28→56→112→224px, outputs `(B, 2, 224, 224)` logits). ImageNet normalization is applied inside `forward()` so tiles can arrive in [0, 1] range as usual. `download_retfound_weights()` fetches `RETFound_oct.pth` from HuggingFace Hub (`iszt/RETFound_mae_natureOCT`) on first use, caching to `~/.cache/retina_painter/`. **This repo is gated — users must request access at https://huggingface.co/iszt/RETFound_mae_natureOCT and authenticate via `huggingface_hub.login()` before the automatic download will work. In practice, use `setup_retfound.py` which downloads from Google Drive instead.**

**Loss** (`loss.py`): RootPainter's combined Dice + 0.3 Cross-Entropy objective,
with untouched pixels masked inside the loss.

**Fine-tuning and optimizer:** The first 21 of 24 encoder blocks are frozen.
The remaining encoder blocks + decoder use AdamW (`lr=1e-4`,
`weight_decay=1e-4`). These settings intentionally match `retfound_rfa` so a
plain-vs-RFA run primarily compares decoder architecture.

### RETFound + RFA-U-Net decoder (`--model-type retfound_rfa`)

Implements the architecture from Hayati et al. (2025) — *RFA-U-Net: A Foundation Model-Driven Approach for Accurate Choroid Segmentation in OCT Imaging* (medRxiv 2025.05.03.25326923). Reference implementation: https://github.com/Alirezahayatimedtech/RFA-U-Net

Defined in **`retfound_rfa_model.py`** — `RETFoundSegRFA`: same RETFound encoder, but uses `forward_multi_features` to capture features at ViT blocks Z6, Z12, Z18, Z24 (indices 5, 11, 17, 23). Each `(B, 196, 1024)` output is reshaped to `(B, 1024, 14, 14)` and fed into a U-Net-style decoder:

| Stage | Input | Skip source | Skip proj | Output |
|---|---|---|---|---|
| d1 | z24, 14² | z18 | 1 upstep → 28², 512ch | 28², 512ch |
| d2 | 28², 512 | z12 | 2 upsteps → 56², 256ch | 56², 256ch |
| d3 | 56², 256 | z6 | 3 upsteps → 112², 128ch | 112², 128ch |
| d4 | 112², 128 | — | — | 224², 64ch |
| out | 224², 64 | — | — | 224², 2 |

Each skip connection passes through an `_AttentionGate` (additive attention: Wg + Ws → ReLU → sigmoid → scale) before concatenation with the upsampled decoder feature.

**Key implementation notes:**
- `_make_skip_pyramid(in_ch, out_ch, up_steps)` — progressive `ConvTranspose2d` chain to upsample 14→28/56/112 with channel reduction
- `freeze_encoder_blocks(num_blocks=21)` — freezes first N transformer blocks; leaves decoder + last 3 blocks trainable (matches RFA-U-Net paper)
- `in_w = out_w = 224`, same tile_pad=0 constraint as `retfound`

**Loss** (`loss.py`): RootPainter's combined Dice + 0.3 Cross-Entropy objective, with untouched pixels masked inside the loss. This is the painter-facing default for every model so model selection does not silently change the optimization objective. Tversky remains available only through an explicit `--loss-type tversky` ablation.

**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4) applied to trainable params only (frozen blocks excluded). Plain `retfound` uses the same optimizer and 21/24-block freezing policy for a controlled decoder comparison.

**RETFound weight notes (shared by both retfound variants):**
- The checkpoint is a full MAE checkpoint (~3.95 GB), not just encoder weights
- `iszt/RETFound_mae_natureOCT` uses `model.safetensors` format on HuggingFace (not `.pth`); the `.pth` file is sourced from Google Drive (file ID: `1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2`) via `setup_retfound.py`
- Loading the checkpoint takes 2–5 minutes on each trainer startup — this is expected due to the file size

**Key difference from retfound:** For both, `in_w = out_w = 224` (no valid-convolution crop). `retfound_rfa` consumes more VRAM due to storing 4 intermediate feature maps; the per-item memory estimate (1.5 GB) is the same conservative value used for plain `retfound`.

## Model Factory Pattern

`model_utils.py` exposes a `_build_model(model_type)` factory used by:
- `load_model(path, model_type='unet')` — loads saved weights into the right architecture
- `create_first_model_with_random_weights(model_dir, model_type='unet')` — for retfound, downloads pretrained encoder weights and initializes a random decoder
- `get_prev_model(model_dir, model_type='unet')` — wraps `load_model`

`Trainer` stores `self.model_type` and passes it through all model-loading calls. The `--model-type` CLI arg in `main.py` sets this.

## Development Setup

```bash
# Trainer
cd trainer
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate (requires RemoteSigned execution policy)
pip install -r requirements.txt   # includes timm>=0.9.0 and huggingface_hub>=0.20.0
pip install pytest

# Painter
cd painter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows notes:**
- Python 3.11 and 3.12 are both supported on Windows (3.13+ is not). Use `py -3.11 -m venv env` or `py -3.12 -m venv env`
- If `env\Scripts\activate` is blocked: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- PyTorch cu124 wheels do not exist for torch>=2.7; use cu126: `pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126` before `pip install -r requirements.txt`. The download is ~2.5 GB and can take 10–20 minutes.
- The `start-trainer` entry point is defined in `trainer/src/__init__.py` (not `main.py`) — both must be kept in sync when adding CLI arguments
- RETFound weights require HuggingFace authentication; call `from huggingface_hub import login; login(token='...')` before first use, or use `setup_retfound.py --gdrive` to download from Google Drive without authentication
- **Windows Smart App Control** may block unsigned CUDA `.dll` and `.pyd` files from PyPI. If you see a `torchvision::nms` error or DLL import failure, run PowerShell **as Administrator** from the repo root and unblock:
  ```powershell
  Get-ChildItem -Path "trainer\env\Lib\site-packages\torch\lib" -Filter "*.dll" | Unblock-File
  Get-ChildItem -Path "trainer\env\Lib\site-packages\torchvision" -Recurse -Filter "*.pyd" | Unblock-File
  ```
  Then restart the trainer. If the venv was created inside a OneDrive-synced folder, move the repo to a local path (e.g. `C:\Users\<user>\Desktop\`) and recreate the venv — OneDrive can corrupt venv Scripts paths.
- **Windows paging file:** The RETFound model (~4 GB VRAM) plus DataLoader workers importing scipy/skimage can exhaust the default paging file. Symptoms: `MemoryError` or `DLL load failed: paging file too small`. Fix: move the paging file to a drive with ≥20 GB free and set Initial: 16384 MB / Maximum: 32768 MB via `sysdm.cpl` → Advanced → Performance → Virtual Memory. Never set a large paging file on a nearly-full drive — it causes system instability.
- **Windows DataLoader workers:** Always use `--maxworkers 4` (or lower) on Windows with RETFound models. The default of 12 workers simultaneously importing large DLLs exhausts virtual memory. Example: `python -u main.py --model-type retfound_rfa --maxworkers 4`
- **Sync directory and disk space:** Store the sync directory on a drive with ample free space. Each RETFound checkpoint is ~1.2 GB; several accumulate during training. Without `--syncdir`, the trainer reads from `~/root_painter_settings.json` (written by the painter on first run). The painter and trainer must point to the same sync directory or no instructions will be exchanged.
- **Virtual environment activation:** Always activate the venv before running the trainer (`env\Scripts\activate`). Without it, the system Python (which lacks CUDA-enabled PyTorch) is used and CUDA will show as unavailable.

## Running

**Always run the trainer from `trainer/src/`** — imports are relative and will fail from any other directory.

```bash
# U-Net mode (default, unchanged from RootPainter)
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync

# RETFound plain decoder
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync --model-type retfound

# RETFound + RFA-U-Net attention decoder (separate architecture comparison)
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync --model-type retfound_rfa

# Optional loss ablation: same RFA architecture with Tversky
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync --model-type retfound_rfa --loss-type tversky

# Or via pip entry point after install
start-trainer --syncdir ~/root_painter_sync
start-trainer --syncdir ~/root_painter_sync --model-type retfound
start-trainer --syncdir ~/root_painter_sync --model-type retfound_rfa

# Painter (unchanged)
cd painter/src/main/python && python main.py
```

Use `-u` (unbuffered) so print statements appear immediately in the terminal.

## Testing

Tests are in `trainer/tests/`. Run from that directory. Full fast trainer suite is 99 tests; runtime depends heavily on the available accelerator.

```bash
cd trainer/tests

# Full unit suite (fast, no downloads)
python -m pytest test_loss.py test_unet.py test_utils.py test_loss_masking.py \
                  test_retfound.py test_retfound_rfa.py \
                  test_fundusegmenter.py test_metrics.py test_instructions.py \
                  test_training_control.py test_nested_annotation_datasets.py -v

# Individual files
python -m pytest test_retfound.py -v          # RETFound plain decoder
python -m pytest test_retfound_rfa.py -v      # RFA decoder + optional Tversky tests
python -m pytest test_loss_masking.py -v      # sparse-supervision masking

# Single test
python -m pytest test_unet.py::TestUNet::test_forward_pass -v

# Training benchmarks (downloads datasets from Zenodo on first run, slow)
python -m pytest test_training.py -v -s
```

**Coverage:**
- `test_retfound.py` (14 tests) — ViT token shape, `RETFoundSeg` forward pass shape, strict checkpoint compatibility, softmax correctness, gradient flow, 21/24-block encoder freezing, and a tiling smoke test.
- `test_retfound_rfa.py` (18 tests) — `forward_multi_features` shape, `RETFoundSegRFA` forward pass shape, softmax correctness, no-NaN, gradient flow, encoder freezing, Tversky loss properties, and a tiling smoke test.
- `test_loss_masking.py` (18 tests, Phase 1 + loss routing) — sparse-supervision regression tests: untouched pixels contribute zero gradient and zero loss-value sensitivity for both `combined_loss` and `tversky_loss`; loss is invariant to the amount of untouched canvas; parity with legacy unmasked loss when mask is all-1s; model families route to their intended objectives; and explicit loss overrides work for controlled ablations.
- `test_training_control.py` (22 tests) — trial seeding, RNG isolation, continuous-loss checkpoint promotion while hard F1 remains zero, background-only validation, deterministic fixed and adaptive correction-class sampling, background-loss-gated provisional UI warm-up, stable checkpoint routing, foreground-gated automatic rollback, rollback-grace restart after annotation updates, and U-Net discarded-border supervision.
- `test_nested_annotation_datasets.py` (2 tests) — reproducible seeded ranking, strict subset nesting, exclusion handling, manifests, copy verification, and refusal to delete unexpected images.

**End-to-end smoke scripts** (not collected by pytest, run manually):
- `smoke_phase1.py` — UNet integration: 30-step training run, legacy-vs-fixed loss comparison across untouched-fraction settings, gradient isolation. Runs in ~30s on CPU.
- `smoke_phase1_gpu.py` — RETFound + RETFound-RFA integration: same checks for both real model architectures. Auto-detects CUDA / MPS / CPU. Use `--use-real-weights` to also load the real RETFound checkpoint (slow). Run on a GPU machine for best results.

## Linting

Pylint config is at `painter/.pylint`. Many rules are intentionally disabled.

```bash
pylint painter/src/main/python/*.py
```

## Build

**Trainer PyPI package:**
```bash
cd trainer && python -m build
```

**Painter executable (PyInstaller):**
```bash
cd painter && python src/build/run_pyinstaller.py
```

**Workstation bundle (trainer + painter):**
```bash
./build_workstation_ubuntu_cuda128.sh              # Linux (RTX 5000 series, default)
./build_workstation_ubuntu_cuda128.sh rtx50        # same as above
./build_workstation_ubuntu_cuda128.sh broad        # Linux (GTX 1660 through RTX 4090)
./build_workstation_win.ps1                        # Windows
./build_workstation_mac.sh                         # macOS
```

**Custom PyTorch wheels (for minimal CUDA workstation builds):**

The Linux CUDA 12.8 workstation uses custom-built PyTorch wheels with unused CUDA libraries stripped out (~330MB vs ~1.5GB). Two variants exist: RTX 5000 (sm_120, `requirements_torch_cu128.txt`) and broad (sm_75/80/86/89, `requirements_torch_cu128_broad.txt`). Wheels are hosted as GitHub release assets.

```bash
# Build wheels locally (requires CUDA 12.8 toolkit + Python 3.11)
./build_custom_torch.sh                         # defaults: v2.7.1 / v0.22.1 / sm 12.0
./build_custom_torch.sh v2.8.0 v0.23.0 "12.0"  # specific versions
MAX_JOBS=16 ./build_custom_torch.sh             # override parallelism
```

Outputs wheels to `./dist/`. After building, upload to a GitHub release and update the URLs in `trainer/requirements_torch_cu128.txt`.

## Key Constraints

- Do not include `Co-Authored-By` or any agent attribution in commit messages
- Python 3.11–3.12 required (`>=3.11,<3.13`)
- Trainer imports are relative (e.g., `from unet import ...`), not package-qualified — tests and entry points run from `trainer/src/`
- Batch size is auto-detected from GPU memory (CUDA/MPS/CPU fallback); retfound uses 1.5 GB/item estimate vs. 3.8 GB/item for unet
- Painter instructions remain filesystem JSON, with optional `model_type`, `model_seed`, and `training_seed` fields for new projects; older instructions remain supported.
- `Trainer.fix_config_paths` must treat `model_type` as an identifier, not a path. If it is path-prefixed, `_build_model` silently falls back to U-Net even when the painter selected a RETFound model.
- `model_type` must be propagated through every model-loading call: `load_model`, `create_first_model_with_random_weights`, `ensemble_segment`, and `get_prev_model`. Omitting it silently loads a UNet instead of RETFound. This applies to all three values: `'unet'`, `'retfound'`, `'retfound_rfa'`.
- `retfound` and `retfound_rfa` produce **incompatible checkpoints** (different decoder state_dict keys). Never load a `retfound` `.pkl` with `--model-type retfound_rfa` or vice versa — it will silently produce a shape mismatch or wrong architecture.
- When `in_w == out_w` (RETFound), `tile_pad = 0`. Guard any annotation crop with `if tile_pad > 0:` — Python's `x[0:-0]` is `x[0:0]` (empty), not a no-op.

## Current Development Status

### Phase 1a: RETFound Backbone — COMPLETE

**Model and infrastructure:**
- `retfound_vit.py`: ViT-Large encoder matching RETFound checkpoint format; `flush=True` on all print statements for real-time terminal output
- `retfound_model.py`: `RETFoundSeg` (encoder + decoder) + weight download helper; weights sourced from Google Drive (`1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2`) via `setup_retfound.py`; HuggingFace fallback available for users with `iszt/RETFound_mae_natureOCT` access
- `main.py`: `--model-type` CLI arg
- `src/__init__.py`: `start()` entry point updated to support `--model-type` (matches `main.py`); stdout/stderr reconfigured with `line_buffering=True` for Windows terminal output
- `trainer.py`: model factory, `in_w=out_w=224` for retfound
- `model_utils.py`: model-type-aware load/create/validate; `ensemble_segment()` accepts and propagates `model_type`; progress prints with `flush=True`
- `trainer/requirements.txt`: added `timm>=0.9.0`, `huggingface_hub>=0.20.0`; pinned `torch==2.8.0+cu126` (cu124 builds unavailable for torch>=2.7)
- `painter/requirements.txt`: updated `scikit-image`, `scipy`, `matplotlib`, `pyqtgraph`, `PyWavelets`, `qimage2ndarray` for Python 3.12 compatibility
- `setup_retfound.py`: interactive weight download helper (Google Drive via `gdown`, or HuggingFace token)
- `test_retfound.py`: 11 passing unit tests

**Bugs fixed:**
- `datasets.py`: `tile_pad = (in_w - out_w) // 2` is 0 for RETFound; `foreground[0:-0]` produces an empty tensor. Fixed with `if tile_pad > 0:` guard — without this, training crashes with `RuntimeError: tensor size (224) must match tensor b (0)`.
- `model_utils.py` / `trainer.py`: `ensemble_segment()` was calling `load_model(path)` without `model_type`, silently loading a UNet architecture in RETFound mode and producing a state_dict mismatch error during segmentation. Fixed by adding `model_type` parameter to `ensemble_segment()` and propagating `self.model_type` from `Trainer.segment_file()`.
- `trainer.py`: `create_first_model_with_random_weights(model_dir)` in `segment()` was missing `model_type=self.model_type`, creating a UNet on first segmentation in RETFound mode. Fixed.

### Phase 1b: RFA-U-Net Attention Decoder — COMPLETE

Based on: Hayati, A. et al. (2025). RFA-U-Net: A Foundation Model-Driven Approach for Accurate Choroid Segmentation in OCT Imaging. medRxiv 2025.05.03.25326923. https://doi.org/10.1101/2025.05.03.25326923

**New and modified files:**
- `retfound_vit.py`: added `forward_multi_features(x, indices)` to expose intermediate ViT block outputs (Z6/Z12/Z18/Z24) for U-Net skip connections without modifying the existing `forward_features` contract
- `retfound_rfa_model.py` (new): `RETFoundSegRFA` — same RETFound encoder + RFA-U-Net decoder. Key components: `_AttentionGate` (additive attention), `_DecoderBlock` (upconv + attention gate + fuse), `_make_skip_pyramid` (14→28/56/112 px with channel reduction), `RETFoundSegRFA.freeze_encoder_blocks(21)` (freezes first 21 of 24 blocks, last 3 + decoder trainable)
- `loss.py`: added `tversky_loss(predictions, labels, alpha=0.7, beta=0.3, class_weights=(1.0, 2.0))`
- `model_utils.py`: `_build_model` and `create_first_model_with_random_weights` handle `retfound_rfa`; reuses `download_retfound_weights()` from `retfound_model`
- `main.py` + `src/__init__.py`: `--model-type` choices extended to `['unet', 'retfound', 'retfound_rfa']`
- `trainer.py`: `retfound_rfa` shares `in_w=out_w=224`; uses the common combined-loss default plus AdamW (lr=1e-4), and calls `freeze_encoder_blocks(21)` after first model load. The original Tversky objective remains available as an explicit ablation.
- `test_retfound_rfa.py` (new): 15 passing unit tests

**Backward compatibility:** `retfound` checkpoints are incompatible with `retfound_rfa` (different decoder keys). Projects must be started fresh with the new `--model-type`. Existing `retfound` projects continue to work unchanged.

### Phase 2: LoRA Fine-Tuning — PLANNED
- Freeze ViT encoder weights; inject LoRA adapter layers into attention blocks
- Only LoRA params + decoder trained per interactive step → fast updates on desktop GPU
- Optimizer may switch from SGD to AdamW with warmup for LoRA

### Phase 3: Curriculum Learning — DROPPED (2026-11-11)
- Removed from scope by project decision. Do not propose or build a staged training scheduler, synthetic-lesion stages, or easy/hard image tagging.

### Phase 4: Multi-Class Segmentation — PLANNED
- Extend decoder output head from 2 channels to N classes
- Update loss to per-class Dice + cross-entropy
- Single model simultaneously segments multiple biomarker types (RIPL + SDD)
- Painter UI changes needed for multi-class overlay colors

### Annotation semantics (sparse supervision policy)

RetinaPainter uses **sparse corrective annotation**: only pixels the clinician
explicitly painted as foreground or background are supervised. Untouched pixels
are unknown and must contribute zero loss and zero gradient. A dedicated
pixel-level `unsure` category with exclusion from loss is planned, not implemented.
The existing scan-level `uncertain` review decision is separate metadata; the
trainer reads red/green annotation PNGs and does not automatically exclude those
painted pixels because of that decision.

**Phase 1 status — DONE (2026-04-27).** The masked loss path is now correct: `combined_loss` and `tversky_loss` accept a `mask` argument and apply it inside the loss; the old `outputs[:, c] *= defined_tiles` trick was leaky (softmax(0,0) = (0.5, 0.5) still incurred a constant CE penalty per untouched pixel). `datasets.py` now uses `np.logical_or(foreground, background)` for the supervision mask, with an assertion that fg/bg never overlap. `model_utils.unet_segment` also got a defensive `cnn = cnn.to(device)` so any caller that forgets to move a `DataParallel`-wrapped model to the inference device doesn't trip a CPU-vs-MPS/CUDA mismatch on RETFound's registered ImageNet-normalization buffers. Regression tests live at `trainer/tests/test_loss_masking.py`; UNet smoke at `trainer/tests/smoke_phase1.py`; RETFound + RETFound-RFA GPU smoke at `trainer/tests/smoke_phase1_gpu.py`. Expected post-fix training loss is lower by an amount that scales roughly linearly with the untouched fraction of each tile — that is **not** a regression. (Measured on a 50%-untouched synthetic tile: combined_loss drops ~0.14 nats, tversky_loss drops ~0.06 nats.)

### Possible Future Direction: Dense Corrected-Target Training
- The current training flow is intended to remain RootPainter-style corrective annotation, where the user labels only errors rather than drawing dense masks.
- A possible future research direction is to keep that sparse user interaction but derive a dense training target by starting from the model's previous segmentation and applying the clinician's edits.
- This is not the current contract and should not be introduced implicitly during bug fixes. Changing from sparse supervision on explicitly corrected pixels to dense corrected-target supervision would be a training-policy change that needs explicit evaluation, especially for tiny retinal biomarkers where missed lesions are easy to reinforce.
