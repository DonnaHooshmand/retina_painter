# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

## Project Overview

**RetinaPainter** is a fork of [RootPainter](https://github.com/Abe404/root_painter) adapted for retinal OCT biomarker detection. It uses a client-server architecture where the **painter** (PyQt5 GUI client) and **trainer** (PyTorch server) communicate via JSON instruction files in a shared filesystem directory (the "sync directory"). No network protocol is used—communication works over local filesystem, sshfs, Dropbox, or Google Drive.

The key departure from RootPainter is the model backend: instead of a U-Net trained from scratch, RetinaPainter uses the **RETFound ViT-Large foundation model** (pre-trained on 1.6M retinal images) as an encoder, with a lightweight convolutional decoder added for pixel-level segmentation. This enables clinically useful models from far fewer labeled examples (~100–200 images vs. thousands).

## Architecture

**Two independent Python applications:**

- **`painter/`** — PyQt5 desktop GUI. Users annotate images with brush strokes, view model predictions as overlays, and manage projects/datasets. Entry point: `painter/src/main/python/main.py`. Main window class: `root_painter.py`. **Unchanged from RootPainter.**
- **`trainer/`** — PyTorch training server. Watches the sync directory for instructions, trains models, performs segmentation. Entry point: `trainer/src/main.py`. Core loop: `trainer.py` (`Trainer.main_loop()`).

**Filesystem-based IPC:** The client writes JSON instruction files to `<syncdir>/instructions/`. The trainer polls for these, processes them (train, segment, etc.), and writes segmentation results back to the project directory. The `instructions.py` module in the painter handles creating these files.

**Workstation mode:** `server_manager.py` in the painter can auto-launch a bundled trainer executable, or in dev mode, launch the trainer from `trainer/env/bin/python`.

**Annotation routing — train vs. validation:** New annotations created in the painter are saved to either `<project>/annotations/train/` or `<project>/annotations/val/` based on a 5:1 file-count ratio (`get_new_annot_target_dir` in [painter/src/main/python/file_utils.py:73](painter/src/main/python/file_utils.py:73)). The router has **no awareness of patient ID or any other grouping** — it routes purely by maintaining the count ratio. For research projects with patient-level data leakage concerns (e.g. retinal OCT, where the same patient's scans look very similar), the default router can scramble an externally-prepared patient-level split, putting the same patient's images in both `train/` and `val/` and making the val-F1 early-stopping signal over-optimistic.

**Reproducible annotation order:** The New Project dialog has an **Image order seed** (default `0`). `create_project.py` sorts the dataset filenames, shuffles them with a local seeded RNG, stores both `image_order_seed` and the resulting `file_names` list in the `.seg_proj`, and navigation follows that stored list. Use the same fixed dataset and seed for matched model trials. This setting controls the B-scans presented to the annotator; a trainer seed does not.

Two workarounds today, with a planned permanent fix:
- **Pre-populate empty annotation PNGs** at the correct train/val location before opening the painter. The painter's `get_annot_path` ([file_utils.py:58](painter/src/main/python/file_utils.py:58)) finds the existing file and `maybe_save_annotation` overwrites in place, never invoking the 5:1 router. The `prepare_annotations.py` script in the user's data-prep tooling does this.
- **Manually move files** between `annotations/train/` and `annotations/val/` after sessions to enforce the desired split.
- **Planned: explicit train/val source folders** — see the `train-val-split` branch. A "New Project" checkbox lets the user point at separate train and val image directories so the painter routes annotations by source folder rather than by count. Eliminates the workaround entirely.

This concern only affects the **internal validation signal during training** (model selection, early stopping). The held-out *test* set is always a separate physical folder outside the painter project, so its integrity is preserved automatically.

## Clinical Evaluation Contract

RetinaPainter's primary RIPL outcome is **B-scan-level detection**, not contour
agreement. A held-out B-scan is positive when the clinician gold standard says
that at least one RIPL is present. Final evaluation must report the raw
TP/FP/FN/TN confusion matrix plus sensitivity, specificity, PPV, NPV, balanced
accuracy, and accuracy. Pixel Dice/IoU are secondary engineering diagnostics
only and must not be the headline basis for choosing an architecture.

Do not confuse the clinical endpoint with the training surrogate. The trainer
currently uses masked pixel losses, masked pixel F1 for checkpoint selection,
and masked soft-Dice loss for early stopping. This remains the current
implementation until a patient-separated detection validation manifest and
evaluator exist. Sparse corrective annotations are not exhaustive and cannot
be used to calculate honest B-scan false negatives or true negatives.

If individual lesions are evaluated later, use one-to-one lesion matching and
report lesion TP/FP/FN, sensitivity, precision, and false positives per B-scan;
lesion-level TN is undefined.

## Models

### U-Net (original, `--model-type unet`, default)

Defined in `unet.py` (`UNetGNRes`). Uses Group Normalization (not Batch Norm) with residual connections. Default input patch size 572×572, output 500×500 (valid convolutions crop 36px per side). Valid patch sizes: 572, 556, 540, ..., 28.

### RETFound plain decoder (`--model-type retfound`)

Defined across two files:

- **`retfound_vit.py`** — `RETFoundViT`: ViT-Large (patch_size=16, embed_dim=1024, depth=24, num_heads=16) with sin-cos positional embeddings. `forward_features(x)` returns `(B, 196, 1024)` patch tokens (cls token dropped). Also exposes `forward_multi_features(x, indices)` which captures intermediate block outputs for RFA skip connections. Weights match the RETFound checkpoint format exactly.
- **`retfound_model.py`** — `RETFoundSeg`: encoder (`RETFoundViT`) + `_SegDecoder` (4-stage ConvTranspose2d upsampler: 14→28→56→112→224px, outputs `(B, 2, 224, 224)` logits). ImageNet normalization is applied inside `forward()` so tiles can arrive in [0, 1] range as usual. `download_retfound_weights()` fetches `RETFound_oct.pth` from HuggingFace Hub (`iszt/RETFound_mae_natureOCT`) on first use, caching to `~/.cache/retina_painter/`. **This repo is gated — users must request access at https://huggingface.co/iszt/RETFound_mae_natureOCT and authenticate via `huggingface_hub.login()` before the automatic download will work. In practice, use `setup_retfound.py` which downloads from Google Drive instead.**

**Loss** (`loss.py`): RootPainter's combined Dice + 0.3 Cross-Entropy objective,
with untouched pixels masked inside the loss.

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

**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4) applied to trainable params only (frozen blocks excluded). `retfound` continues to use SGD.

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

# RETFound + RFA-U-Net attention decoder (recommended for new projects)
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

Tests are in `trainer/tests/`. Run from that directory. Full unit suite is 67 tests; runtime depends heavily on the available accelerator.

```bash
cd trainer/tests

# Full unit suite (fast, no downloads)
python -m pytest test_loss.py test_unet.py test_utils.py test_loss_masking.py \
                  test_retfound.py test_retfound_rfa.py \
                  test_fundusegmenter.py test_metrics.py test_instructions.py -v

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
- `test_retfound.py` (12 tests) — ViT token shape, `RETFoundSeg` forward pass shape, softmax correctness, gradient flow through decoder, and a tiling smoke test.
- `test_retfound_rfa.py` (18 tests) — `forward_multi_features` shape, `RETFoundSegRFA` forward pass shape, softmax correctness, no-NaN, gradient flow, encoder freezing, Tversky loss properties, and a tiling smoke test.
- `test_loss_masking.py` (18 tests, Phase 1 + loss routing) — sparse-supervision regression tests: untouched pixels contribute zero gradient and zero loss-value sensitivity for both `combined_loss` and `tversky_loss`; loss is invariant to the amount of untouched canvas; parity with legacy unmasked loss when mask is all-1s; model families route to their intended objectives; and explicit loss overrides work for controlled ablations.

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
- The painter and JSON instruction format are **unchanged** — all RetinaPainter changes are trainer-side only
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

### Phase 3: Curriculum Learning — PLANNED
- Staged training scheduler: synthetic lesions → easy real cases → hard cases → confounders
- User labels certain images as "easy" or "hard"; system schedules training order accordingly
- Evaluate on RIPL and SDD detection tasks

### Phase 4: Multi-Class Segmentation — PLANNED
- Extend decoder output head from 2 channels to N classes
- Update loss to per-class Dice + cross-entropy
- Single model simultaneously segments multiple biomarker types (RIPL + SDD)
- Painter UI changes needed for multi-class overlay colors

### Annotation semantics (sparse supervision policy)

RetinaPainter uses **sparse corrective annotation**: only pixels the clinician explicitly painted as foreground or background are supervised. Untouched pixels are unknown — they must contribute zero loss and zero gradient. A forthcoming `unsure` annotation category will also be excluded from loss while retaining ambiguity metadata.

**Phase 1 status — DONE (2026-04-27).** The masked loss path is now correct: `combined_loss` and `tversky_loss` accept a `mask` argument and apply it inside the loss; the old `outputs[:, c] *= defined_tiles` trick was leaky (softmax(0,0) = (0.5, 0.5) still incurred a constant CE penalty per untouched pixel). `datasets.py` now uses `np.logical_or(foreground, background)` for the supervision mask, with an assertion that fg/bg never overlap. `model_utils.unet_segment` also got a defensive `cnn = cnn.to(device)` so any caller that forgets to move a `DataParallel`-wrapped model to the inference device doesn't trip a CPU-vs-MPS/CUDA mismatch on RETFound's registered ImageNet-normalization buffers. Regression tests live at `trainer/tests/test_loss_masking.py`; UNet smoke at `trainer/tests/smoke_phase1.py`; RETFound + RETFound-RFA GPU smoke at `trainer/tests/smoke_phase1_gpu.py`. Expected post-fix training loss is lower by an amount that scales roughly linearly with the untouched fraction of each tile — that is **not** a regression. (Measured on a 50%-untouched synthetic tile: combined_loss drops ~0.14 nats, tversky_loss drops ~0.06 nats.)

### Possible Future Direction: Dense Corrected-Target Training
- The current training flow is intended to remain RootPainter-style corrective annotation, where the user labels only errors rather than drawing dense masks.
- A possible future research direction is to keep that sparse user interaction but derive a dense training target by starting from the model's previous segmentation and applying the clinician's edits.
- This is not the current contract and should not be introduced implicitly during bug fixes. Changing from sparse supervision on explicitly corrected pixels to dense corrected-target supervision would be a training-policy change that needs explicit evaluation, especially for tiny retinal biomarkers where missed lesions are easy to reinforce.
