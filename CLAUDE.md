# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RetinaPainter** is a fork of [RootPainter](https://github.com/Abe404/root_painter) adapted for retinal OCT biomarker detection. It uses a client-server architecture where the **painter** (PyQt5 GUI client) and **trainer** (PyTorch server) communicate via JSON instruction files in a shared filesystem directory (the "sync directory"). No network protocol is used—communication works over local filesystem, sshfs, Dropbox, or Google Drive.

The key departure from RootPainter is the model backend: instead of a U-Net trained from scratch, RetinaPainter uses the **RETFound ViT-Large foundation model** (pre-trained on 1.6M retinal images) as an encoder, with a lightweight convolutional decoder added for pixel-level segmentation. This enables clinically useful models from far fewer labeled examples (~100–200 images vs. thousands).

## Architecture

**Two independent Python applications:**

- **`painter/`** — PyQt5 desktop GUI. Users annotate images with brush strokes, view model predictions as overlays, and manage projects/datasets. Entry point: `painter/src/main/python/main.py`. Main window class: `root_painter.py`. **Unchanged from RootPainter.**
- **`trainer/`** — PyTorch training server. Watches the sync directory for instructions, trains models, performs segmentation. Entry point: `trainer/src/main.py`. Core loop: `trainer.py` (`Trainer.main_loop()`).

**Filesystem-based IPC:** The client writes JSON instruction files to `<syncdir>/instructions/`. The trainer polls for these, processes them (train, segment, etc.), and writes segmentation results back to the project directory. The `instructions.py` module in the painter handles creating these files.

**Workstation mode:** `server_manager.py` in the painter can auto-launch a bundled trainer executable, or in dev mode, launch the trainer from `trainer/env/bin/python`.

## Models

### U-Net (original, `--model-type unet`, default)

Defined in `unet.py` (`UNetGNRes`). Uses Group Normalization (not Batch Norm) with residual connections. Default input patch size 572×572, output 500×500 (valid convolutions crop 36px per side). Valid patch sizes: 572, 556, 540, ..., 28.

### RETFound ViT-Large (`--model-type retfound`)

Defined across two files:

- **`retfound_vit.py`** — `RETFoundViT`: ViT-Large (patch_size=16, embed_dim=1024, depth=24, num_heads=16) with sin-cos positional embeddings. `forward_features(x)` returns `(B, 196, 1024)` patch tokens (cls token dropped). Weights match the RETFound checkpoint format exactly.
- **`retfound_model.py`** — `RETFoundSeg`: encoder (`RETFoundViT`) + `_SegDecoder` (4-stage ConvTranspose2d upsampler: 14→28→56→112→224px, outputs `(B, 2, 224, 224)` logits). ImageNet normalization is applied inside `forward()` so tiles can arrive in [0, 1] range as usual. `download_retfound_weights()` fetches `RETFound_oct.pth` from HuggingFace Hub (`iszt/RETFound_mae_natureOCT`) on first use, caching to `~/.cache/retina_painter/`. **This repo is gated — users must request access at https://huggingface.co/iszt/RETFound_mae_natureOCT and authenticate via `huggingface_hub.login()` before the automatic download will work. In practice, use `setup_retfound.py` which downloads from Google Drive instead.**

**RETFound weight notes:**
- The checkpoint is a full MAE checkpoint (~3.95 GB), not just encoder weights
- `iszt/RETFound_mae_natureOCT` uses `model.safetensors` format on HuggingFace (not `.pth`); the `.pth` file is sourced from Google Drive (file ID: `1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2`) via `setup_retfound.py`
- Loading the checkpoint takes 2–5 minutes on each trainer startup — this is expected due to the file size

**Key difference:** For retfound, `in_w = out_w = 224` (no valid-convolution crop). The patch-size assertion in `Trainer.__init__` is skipped, and the per-item memory estimate is 1.5 GB (ViT-Large is heavier than U-Net).

**Loss** (`loss.py`): Combined 0.7 Dice + 0.3 Cross-Entropy with softmax over 2 channels (foreground/background). Unchanged for both model types.

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
- Python 3.12 is required on Windows (3.11 no longer has binary installers; use `py -3.12 -m venv env`)
- If `env\Scripts\activate` is blocked: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- PyTorch cu124 wheels do not exist for torch>=2.7; use cu126: `pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126` before `pip install -r requirements.txt`
- The `start-trainer` entry point is defined in `trainer/src/__init__.py` (not `main.py`) — both must be kept in sync when adding CLI arguments
- RETFound weights require HuggingFace authentication; call `from huggingface_hub import login; login(token='...')` before first use, or use `setup_retfound.py --gdrive` to download from Google Drive without authentication
- **Windows Smart App Control** may block unsigned CUDA `.dll` and `.pyd` files from PyPI. If you see a `torchvision::nms` error or DLL import failure, run PowerShell **as Administrator** from the repo root and unblock:
  ```powershell
  Get-ChildItem -Path "trainer\env\Lib\site-packages\torch\lib" -Filter "*.dll" | Unblock-File
  Get-ChildItem -Path "trainer\env\Lib\site-packages\torchvision" -Recurse -Filter "*.pyd" | Unblock-File
  ```
  Then restart the trainer. If the venv was created inside a OneDrive-synced folder, move the repo to a local path (e.g. `C:\Users\<user>\Desktop\`) and recreate the venv — OneDrive can corrupt venv Scripts paths.

## Running

**Always run the trainer from `trainer/src/`** — imports are relative and will fail from any other directory.

```bash
# U-Net mode (default, unchanged from RootPainter)
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync

# RETFound mode
cd trainer/src && python -u main.py --syncdir ~/root_painter_sync --model-type retfound

# Or via pip entry point after install
start-trainer --syncdir ~/root_painter_sync
start-trainer --syncdir ~/root_painter_sync --model-type retfound

# Painter (unchanged)
cd painter/src/main/python && python main.py
```

Use `-u` (unbuffered) so print statements appear immediately in the terminal.

## Testing

Tests are in `trainer/tests/`. Run from that directory:

```bash
# RETFound backbone tests (fast, no download — uses random weights)
cd trainer/tests
python -m pytest test_retfound.py -v

# Original unit tests (fast, no downloads)
python -m pytest test_loss.py test_unet.py test_utils.py -v

# Single test
python -m pytest test_unet.py::TestUNet::test_forward_pass -v

# Training benchmarks (downloads datasets from Zenodo on first run, slow)
python -m pytest test_training.py -v -s
```

`test_retfound.py` covers: ViT token shape, `RETFoundSeg` forward pass shape, softmax correctness, gradient flow through decoder, and a tiling smoke test (skipped if scikit-image not installed).

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

- Do not include `Co-Authored-By` or any Claude attribution in commit messages
- Python 3.11–3.12 required (`>=3.11,<3.13`)
- Trainer imports are relative (e.g., `from unet import ...`), not package-qualified — tests and entry points run from `trainer/src/`
- Batch size is auto-detected from GPU memory (CUDA/MPS/CPU fallback); retfound uses 1.5 GB/item estimate vs. 3.8 GB/item for unet
- The painter and JSON instruction format are **unchanged** — all RetinaPainter changes are trainer-side only
- `model_type` must be propagated through every model-loading call: `load_model`, `create_first_model_with_random_weights`, `ensemble_segment`, and `get_prev_model`. Omitting it silently loads a UNet instead of RETFound.
- When `in_w == out_w` (RETFound), `tile_pad = 0`. Guard any annotation crop with `if tile_pad > 0:` — Python's `x[0:-0]` is `x[0:0]` (empty), not a no-op.

## Current Development Status

### Phase 1: RETFound Backbone — COMPLETE

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
