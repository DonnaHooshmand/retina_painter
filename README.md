## RetinaPainter

Interactive annotation platform for training segmentation models on retinal OCT images with very few labels. A fork of [RootPainter](https://github.com/Abe404/root_painter) that swaps the scratch-trained U-Net for a [RETFound](https://github.com/rmaphoh/RETFound_MAE) ViT-Large foundation model encoder, so clinically useful models can be trained from ~100–200 annotated images instead of thousands. The clinician paints corrections on the model's predictions and the model retrains in real time.

---

## Getting started

### 1. Download RETFound weights (one time, ~3.95 GB)

```bash
python setup_retfound.py
```

Caches `RETFound_oct.pth` to `~/.cache/retina_painter/`. If you have HuggingFace access for `iszt/RETFound_mae_natureOCT`, pass `--token YOUR_HF_TOKEN`.

### 2. Install the trainer (server)

**Mac / Linux:**

```bash
cd trainer
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
cd trainer
py -3.11 -m venv env                      # or py -3.12 — Python 3.13+ not supported
env\Scripts\activate                      # if blocked: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

PyTorch cu126 (~2.5 GB) takes 10–20 minutes; cu124 wheels don't exist for torch≥2.7. See [Windows troubleshooting](#windows-troubleshooting) if installation hangs or crashes.

### 3. Install the painter (client)

**Mac / Linux:**

```bash
cd painter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
cd painter
py -3.11 -m venv env
env\Scripts\activate
pip install -r requirements.txt
```

### 4. Run

Open two terminals — one for the trainer, one for the painter.

**Trainer (always from `trainer/src/`** — imports are relative):

```bash
# Mac / Linux
cd trainer/src && python -u main.py --syncdir /path/to/sync_dir

# Windows (must add --maxworkers 4 to avoid paging-file exhaustion)
cd trainer\src
python -u main.py --syncdir D:\RetinaPainterSync --maxworkers 4
```

**Painter:**

```bash
cd painter/src/main/python
python main.py
```

Point the painter at the same sync directory you passed to the trainer. On first launch, the painter writes that choice to `~/retina_painter_settings.json`.

**First-run wait:** loading the 3.95 GB RETFound checkpoint takes 2–5 minutes on each trainer startup, and again before the first segmentation. Progress prints to the terminal.

**Model type** is selected per project in the painter's New Project dialog (U-Net / RETFound + plain decoder / RETFound + RFA-U-Net). The choice is saved in the project file and forwarded to the trainer automatically.

---

## How it works

### Architecture

Two independent Python applications talking through the filesystem:

- **Painter** (`painter/`, PyQt5) — desktop GUI. Users open images, paint corrections on top of the model's segmentation overlay, and manage projects.
- **Trainer** (`trainer/`, PyTorch) — server. Watches the sync directory, trains models when instructed, runs segmentation on demand, writes results back.

The two communicate by writing JSON instruction files to `<syncdir>/instructions/`. No network protocol — the sync directory can be a local folder, sshfs, Dropbox, or Google Drive. This means the painter can run on a clinical workstation while the trainer runs on a GPU server, with cloud-storage as the only link.

### Models

| Mode | Encoder | Decoder | Loss | Notes |
|---|---|---|---|---|
| `unet` | scratch U-Net (Group Norm + residual) | — | Dice + 0.3 CE | Original RootPainter; 572×572 in / 500×500 out. |
| `retfound` | RETFound ViT-Large (frozen pretraining) | 4-stage ConvTranspose2d | Dice + 0.3 CE | Plain decoder; 224×224 tiles. |
| `retfound_rfa` (recommended) | RETFound ViT-Large | RFA-U-Net (skips Z6/Z12/Z18/Z24 + attention gates) | Tversky (α=0.7, β=0.3) | Freezes first 21 of 24 ViT blocks, trains last 3 + decoder with AdamW. Reaches Dice 95.04 / Jaccard 90.59 on choroid (Hayati et al., 2025). |

The RETFound encoder is pretrained via masked autoencoder self-supervision on 1.6 million unlabeled retinal images, so it already knows what retinas look like. Training only has to learn the lesion-specific decision boundary, which is why ~100–200 labels are enough.

### Annotation semantics — sparse corrective supervision

Pixels the clinician explicitly paints as foreground or background are supervised with a 0/1 label. Untouched pixels are treated as **unknown** and excluded from the loss with zero gradient and zero loss-value contribution. This matters because earlier RootPainter code zeroed the logits at untouched pixels but didn't actually exclude them — every untouched pixel still incurred a constant CE penalty. RetinaPainter's masked loss fixes this; details and tests in [docs/supervision_plan.md](docs/supervision_plan.md).

A future `unsure` annotation category is planned for explicitly ambiguous regions (blurry imagery, hard lesion edges) — also masked from loss but retained as metadata for curriculum learning and uncertainty evaluation. A dense corrected-target alternative (treating untouched as "model was right") remains a possible future research direction but is not the current contract.

### Background — RootPainter and prior OCT work

[RootPainter](https://github.com/Abe404/root_painter) ([Smith et al., New Phytologist 2022](https://doi.org/10.1111/nph.18387)) showed that a few hours of corrective annotation can match fully supervised performance, but it trains a U-Net from scratch on each new task. [Drakopoulos et al. (2024)](https://doi.org/10.3928/23258160-20240410-01) applied RootPainter to retinal OCT and reached ~90% / ~92% accuracy on RIPL and SDD detection from ~6 hours of annotation. RetinaPainter pushes further by combining the same corrective workflow with foundation-model pretraining, so the encoder no longer has to be learned from scratch every time.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1a | RETFound ViT-Large backbone + plain decoder | Complete |
| 1b | RFA-U-Net attention decoder | Complete |
| 1c | Model type UI dropdown, RetinaPainter rename | Complete |
| 1d | Sparse-supervision masking fix (untouched pixels truly excluded from loss) | Complete |
| 1e | Pre-split train/val folder option (avoids 5:1 router scrambling patient-level splits) | In progress (Slice 1: dialog UI shipped) |
| 2a | `unsure` annotation category (painter brush + masked from loss + curriculum metadata) | Planned |
| 2b | LoRA parameter-efficient fine-tuning | Planned |
| 3 | Curriculum learning scheduler (driven in part by `unsure` density) | Planned |
| 4 | Multi-class segmentation support | Planned |
| 5 | Evaluate dense corrected-target training from sparse edits | Possible future work |

---

## Testing

```bash
cd trainer/tests
python -m pytest test_loss.py test_unet.py test_utils.py test_loss_masking.py \
                  test_retfound.py test_retfound_rfa.py -v
```

Expected: 44 passed. Per-file coverage and end-to-end smoke scripts are documented in [trainer/tests/readme.md](trainer/tests/readme.md).

---

## Windows troubleshooting

Most Windows pain comes from CUDA DLLs being blocked by Smart App Control or the system paging file being too small for the RETFound model + DataLoader workers.

- **`torchvision::nms` or DLL load error on first run:** Smart App Control blocked the CUDA DLLs. Run PowerShell **as Administrator** and unblock:
  ```powershell
  Get-ChildItem -Path "env\Lib\site-packages\torch\lib" -Filter "*.dll" | Unblock-File
  Get-ChildItem -Path "env\Lib\site-packages\torchvision" -Recurse -Filter "*.pyd" | Unblock-File
  ```
- **`MemoryError` or "DLL load failed: paging file too small":** the default Windows paging file is too small for the RETFound model (~4 GB VRAM) plus DataLoader workers importing scipy/skimage. Move the paging file to a drive with ≥20 GB free; set Initial 16384 MB / Maximum 32768 MB via `sysdm.cpl` → Advanced → Performance → Virtual Memory. Never set a large paging file on a nearly-full drive — the system becomes unstable.
- **Always pass `--maxworkers 4`** (or lower) on Windows. The default of 12 workers each importing large CUDA DLLs can exhaust virtual memory.
- **OneDrive-synced repo paths can corrupt the venv.** Move the repo to a local path like `C:\Users\<you>\Desktop\` and recreate the venv if `env\Scripts\activate` produces strange path errors.

---

## References

- Zhou, K. et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*, 622, 156–163.
- Hayati, A. et al. (2025). RFA-U-Net: A Foundation Model-Driven Approach for Accurate Choroid Segmentation in OCT Imaging. *medRxiv* 2025.05.03.25326923. https://doi.org/10.1101/2025.05.03.25326923
- Drakopoulos, M. et al. (2024). Machine teaching allows for rapid development of automated systems for retinal lesion detection from small image datasets. *Ophthalmic Surgery, Lasers & Imaging Retina*, 55(8), 475–478.
- Smith, A.G. et al. (2022). RootPainter: deep learning segmentation of biological images with corrective annotation. *New Phytologist*, 236(2), 774–791.

---

## Contributions

Research fork under active development. Open an issue or discussion before submitting a pull request.
