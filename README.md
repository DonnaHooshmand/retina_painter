## RetinaPainter

RetinaPainter is a next-generation interactive annotation platform for training deep learning models on retinal OCT images with minimal labeled data. It extends [RootPainter](https://github.com/Abe404/root_painter) by replacing the scratch-trained U-Net with a [RETFound](https://github.com/rmaphoh/RETFound_MAE) Vision Transformer foundation model backbone, enabling clinically useful segmentation models for novel retinal biomarkers from as few as 100–200 annotated images.

The system retains RootPainter's corrective annotation loop — the clinician paints corrections on the model's predictions and the model retrains in real time — but leverages rich representations pre-trained on 1.6 million retinal images to dramatically reduce the annotation burden.

### Background: RootPainter

RetinaPainter is a fork of [RootPainter](https://github.com/Abe404/root_painter), an open-source GUI tool for segmenting biological images via human-in-the-loop training ([Smith et al., New Phytologist 2022](https://doi.org/10.1111/nph.18387)). RootPainter demonstrated that a few hours of corrective annotation can match fully supervised performance — but it trains a U-Net from scratch on each new task, limiting generalization and requiring dense supervision.

RetinaPainter builds on a prior application of RootPainter to retinal OCT: in [Drakopoulos et al. (2024)](https://doi.org/10.3928/23258160-20240410-01), the interactive loop was used to train the first automated detector for retinal ischemic perivascular lesions (RIPLs) and subretinal drusenoid deposits (SDDs), achieving ~90% and ~92% accuracy respectively with only ~6 hours of annotation time. RetinaPainter aims to push further by combining that corrective paradigm with modern foundation model pretraining.

### What's New in RetinaPainter

- **Foundation model backbone** — The U-Net is replaced with RETFound ViT-Large, a Vision Transformer pre-trained via masked autoencoder self-supervision on 1.6M unlabeled retinal images. Two segmentation heads are available, selected from a dropdown in the **New Project** dialog:
  - **U-Net (original RootPainter)** — scratch-trained U-Net with Group Normalization, unchanged from RootPainter.
  - **RETFound + plain decoder** — plain 4-stage transposed-convolution decoder on top of the RETFound ViT encoder. It uses combined Dice/CE, freezes the first 21 of 24 encoder blocks, and trains the remaining blocks + decoder with AdamW.
  - **RETFound + RFA-U-Net (recommended)** — **RFA-U-Net decoder**: uses four intermediate ViT feature maps (Z6, Z12, Z18, Z24) as U-Net-style skip connections, each passed through a progressive upsampling pyramid and fused via additive attention gates. Achieves Dice 95.04% / Jaccard 90.59% on choroid segmentation vs. all CNN and SOTA baselines (Hayati et al., 2025). Uses the same combined Dice/CE loss as every other painter-selectable model, freezes the first 21 of 24 encoder blocks, and trains the remaining blocks + decoder with AdamW.

- **Parameter-efficient fine-tuning (planned)** — LoRA adapter layers will be injected into the ViT encoder so that each interactive training step updates only a small fraction of parameters, enabling near-real-time model updates on a desktop GPU.

- **Multi-class segmentation (planned)** — A single model will simultaneously label multiple lesion types (e.g. RIPL + SDD), addressing RootPainter's one-class-per-model limitation.

- **Curriculum learning (planned)** — A staged training scheduler will present examples in order of difficulty (synthetic lesions → clear real cases → ambiguous cases → confounders), further reducing the labeled data required to reach clinical accuracy.

- **Annotation semantics: sparse corrective supervision** — Pixels the clinician paints as foreground or background are supervised; untouched pixels are treated as unknown and excluded from the loss with zero gradient and zero loss-value contribution. Earlier RootPainter code zeroed the logits at untouched pixels but did not actually exclude them from loss—every untouched pixel added a constant CE penalty, slowing training. RetinaPainter's masked loss fixes this. A future `unsure` annotation category is planned for explicitly ambiguous regions (blurry imagery, hard lesion edges)—also masked from loss but retained as metadata for curriculum learning and uncertainty evaluation. A dense corrected-target alternative remains a possible future research direction but is not the current contract.

- **Reproducible model trials** — The New Project **Trial seed** fixes image order, the filename-level 5:1 train/validation split, random model/decoder initialization, and training-data RNGs. New projects store the split in the `.seg_proj`, so blank scans or model-dependent corrections cannot shift later images between train and validation. The automatic split is per-file, not patient-aware; use an externally prepared patient-level split for research evaluation.

- **Rare-lesion checkpoint control** — UI checkpoints and early stopping use the continuous masked combined Dice + 0.3 CE objective. Hard pixel F1 remains diagnostic, but cannot leave the UI stuck on a random fuzzy checkpoint while RIPL probabilities are improving below 0.5. Background-only validation uses CE and emits a warning.

- **Detection-first clinical evaluation** — The primary RIPL endpoint is whether a held-out OCT B-scan contains at least one RIPL. Models are compared using the B-scan confusion matrix and sensitivity, specificity, PPV, NPV, and balanced accuracy. Pixel Dice/IoU are secondary training and localization diagnostics, not measures of clinical success.

### Roadmap

| Phase | Description | Status |
|---|---|---|
| 1a | RETFound ViT-Large backbone + plain decoder | Complete |
| 1b | RFA-U-Net attention decoder | Complete |
| 1c | Model type UI dropdown, RetinaPainter rename | Complete |
| 1d | Sparse-supervision masking fix (untouched pixels truly excluded from loss) | Complete |
| 1e | B-scan detection evaluator, threshold calibration, and patient-grouped metrics | Planned — next |
| 2a | `unsure` annotation category (painter brush + masked from loss + curriculum metadata) | Planned |
| 2b | LoRA parameter-efficient fine-tuning | Planned |
| 3 | Curriculum learning scheduler (driven in part by `unsure` density) | Planned |
| 4 | Multi-class segmentation support | Planned |
| 5 | Evaluate dense corrected-target training from sparse edits | Possible future work |

---

### Installation

#### Step 1 — Download RETFound weights

Before running in RETFound mode, download the pretrained weights (~3.95 GB) using the setup helper:

```bash
python setup_retfound.py
```

This downloads `RETFound_oct.pth` from Google Drive to `~/.cache/retina_painter/`. The download happens only once. Run this from the repo root inside the trainer virtual environment (see below).

If you have HuggingFace access approval for `iszt/RETFound_mae_natureOCT`, you can supply a token instead:

```bash
python setup_retfound.py --token YOUR_HF_TOKEN
```

#### Step 2 — Trainer (server)

```bash
cd trainer
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

> **Windows users:** Python 3.11 or 3.12 is supported (3.13+ is not). Use `py -3.11 -m venv env` or `py -3.12 -m venv env` depending on what is installed. Activate with `env\Scripts\activate`. If activation is blocked, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

> **PyTorch on Windows/Linux (CUDA):** Install PyTorch cu126 before the rest of requirements, since cu124 wheels do not exist for torch>=2.7. The download is ~2.5 GB and can take 10–20 minutes depending on your connection:
> ```
> pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
> pip install -r requirements.txt
> ```

> **Windows Smart App Control:** If you see a `torchvision::nms` or DLL error on first run, Windows may have blocked the CUDA DLLs. Run PowerShell **as Administrator** from the venv folder and unblock them:
> ```powershell
> Get-ChildItem -Path "env\Lib\site-packages\torch\lib" -Filter "*.dll" | Unblock-File
> Get-ChildItem -Path "env\Lib\site-packages\torchvision" -Recurse -Filter "*.pyd" | Unblock-File
> ```
> Then restart the trainer.

> **Windows virtual memory (paging file):** The RETFound model (~4 GB VRAM) combined with multiple DataLoader workers can exhaust Windows' default paging file. Symptoms are `MemoryError` or `DLL load failed: paging file too small` during training. Fix: move the paging file to a drive with ample free space (e.g. a data drive), and set it to Initial: 16384 MB / Maximum: 32768 MB via `sysdm.cpl` → Advanced → Performance → Virtual Memory. **Do not set a large paging file on a drive with less than 20 GB free** — this will make the system unstable.

> **Windows DataLoader workers:** Use `--maxworkers 4` (or lower) when running on Windows to avoid paging file exhaustion during training. The default of 12 workers each importing scipy/skimage simultaneously can exceed available virtual memory:
> ```
> python -u main.py --syncdir <path> --maxworkers 4
> ```

#### Step 3 — Painter (client)

```bash
cd painter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

### Running

**Always run the trainer from `trainer/src/`**, not from `trainer/`. The imports are relative and only work from that directory.

#### Start the trainer

```bash
cd trainer/src
python -u main.py --syncdir /path/to/sync_dir
```

On Windows, add `--maxworkers 4` to avoid paging file exhaustion:

```bash
cd trainer/src
python -u main.py --syncdir D:\RootPainterSync --maxworkers 4
```

Or, after `pip install -e .` from the `trainer/` directory:

```bash
start-trainer --syncdir /path/to/sync_dir
```

**Model type is selected per project in the painter UI** — when you create a new project in the painter, a dropdown lets you choose between U-Net, RETFound + plain decoder, or RETFound + RFA-U-Net (recommended). The choice is saved in the project file and automatically sent to the trainer on each train/segment instruction. You do not need to pass `--model-type` on the command line.

The `--model-type` CLI arg still exists as a server-side default (useful when the trainer is started independently and the painter has not yet sent a project instruction), but it is not normally needed.

**Checkpoint selection and early stopping:** both use continuous masked combined Dice + 0.3 CE. The default patience is 60 epochs; raise it for hard, slow-to-converge biomarkers with `--max-epochs-without-progress N` (e.g. `--max-epochs-without-progress 120`). Hard pixel F1 is logged as a diagnostic but cannot block UI checkpoint updates. These are internal training mechanisms, not the clinical endpoint. Clinical model comparison uses patient-separated B-scan RIPL detection.

**`retfound_rfa` vs `retfound`:** Both use the same encoder weights, 224×224 tiles, combined Dice/CE loss, 21/24-block freezing policy, and AdamW settings. `retfound_rfa` adds a U-Net decoder with skip connections from four intermediate ViT layers and attention gates. Keeping the training policy shared makes the comparison primarily a decoder comparison; RFA costs slightly more memory.

**Loss behavior:** `--loss-type auto` is the front-end default and resolves to the inherited combined Dice/CE loss for every model. Choosing a model in the New Project dialog therefore changes the architecture without silently changing the loss. Tversky remains available only as an explicit developer-side ablation through `--loss-type tversky`; use a separate fresh project for it.

**Matched model trials:** The New Project dialog includes a **Trial seed** (default `0`). With the same fixed dataset, it reproduces navigation order, filename-level train/validation membership, model/decoder initialization, and training RNGs for U-Net, RETFound, and RFA trials. The painter sends it to the trainer automatically; no trainer flag is required.

**First-run note:** Loading the RETFound checkpoint (~3.95 GB) takes 2–5 minutes on both backends. Progress prints appear in the terminal. The first segmentation after startup also loads the model from disk, so expect a similar wait before the first overlay appears.

**Sync directory:** The painter and trainer must use the same sync directory. On first launch, the painter will ask where to create the sync directory and writes that choice to `~/retina_painter_settings.json`. Store the sync directory on a drive with plenty of free space — each RETFound checkpoint is ~1.2 GB and several accumulate during training.

#### Painter (client)

```bash
cd painter/src/main/python
python main.py
```

When prompted, point the painter at the same sync directory used when starting the trainer.

---

### Testing

```bash
# RETFound plain decoder tests (no weight download needed)
cd trainer/tests
python -m pytest test_retfound.py -v

# RFA-U-Net decoder + optional Tversky-loss tests (no weight download needed)
python -m pytest test_retfound_rfa.py -v

# FunduSegmenter placeholder + metrics tests
python -m pytest test_fundusegmenter.py test_metrics.py -v

# Reproducible split/seed, checkpoint promotion, and U-Net crop controls
python -m pytest test_training_control.py -v

# Existing unit tests (loss, unet, utilities)
python -m pytest test_loss.py test_unet.py test_utils.py -v
```

`test_retfound_rfa.py` (18 tests) covers:
- `forward_multi_features` returns 4 skip tensors of shape `(B, 196, 1024)`
- `RETFoundSegRFA` forward pass: `(B, 3, 224, 224)` → `(B, 2, 224, 224)`
- Softmax sums to 1, no NaNs, gradients flow to decoder
- `freeze_encoder_blocks(21)` correctly freezes first 21 blocks and leaves decoder trainable
- Tversky loss: scalar output, non-negative, near-zero on perfect predictions, gradients flow
- Tiling smoke test: 512×512 image tiles correctly to `(512, 512)` output

---

### References

- Zhou, K. et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*, 622, 156–163.
- Hayati, A. et al. (2025). RFA-U-Net: A Foundation Model-Driven Approach for Accurate Choroid Segmentation in OCT Imaging. *medRxiv* 2025.05.03.25326923. https://doi.org/10.1101/2025.05.03.25326923
- Drakopoulos, M. et al. (2024). Machine teaching allows for rapid development of automated systems for retinal lesion detection from small image datasets. *Ophthalmic Surgery, Lasers & Imaging Retina*, 55(8), 475–478.
- Smith, A.G. et al. (2022). RootPainter: deep learning segmentation of biological images with corrective annotation. *New Phytologist*, 236(2), 774–791.

---

### Contributions

This is a research fork under active development. Please open an issue or discussion before submitting a pull request.
