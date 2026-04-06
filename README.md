## RetinaPainter

RetinaPainter is a next-generation interactive annotation platform for training deep learning models on retinal OCT images with minimal labeled data. It extends [RootPainter](https://github.com/Abe404/root_painter) by replacing the scratch-trained U-Net with a [RETFound](https://github.com/rmaphoh/RETFound_MAE) Vision Transformer foundation model backbone, enabling clinically useful segmentation models for novel retinal biomarkers from as few as 100–200 annotated images.

The system retains RootPainter's corrective annotation loop — the clinician paints corrections on the model's predictions and the model retrains in real time — but leverages rich representations pre-trained on 1.6 million retinal images to dramatically reduce the annotation burden.

### Background: RootPainter

RetinaPainter is a fork of [RootPainter](https://github.com/Abe404/root_painter), an open-source GUI tool for segmenting biological images via human-in-the-loop training ([Smith et al., New Phytologist 2022](https://doi.org/10.1111/nph.18387)). RootPainter demonstrated that a few hours of corrective annotation can match fully supervised performance — but it trains a U-Net from scratch on each new task, limiting generalization and requiring dense supervision.

RetinaPainter builds on a prior application of RootPainter to retinal OCT: in [Drakopoulos et al. (2024)](https://doi.org/10.3928/23258160-20240410-01), the interactive loop was used to train the first automated detector for retinal ischemic perivascular lesions (RIPLs) and subretinal drusenoid deposits (SDDs), achieving ~90% and ~92% accuracy respectively with only ~6 hours of annotation time. RetinaPainter aims to push further by combining that corrective paradigm with modern foundation model pretraining.

### What's New in RetinaPainter

- **Foundation model backbone** — The U-Net is replaced with RETFound ViT-Large, a Vision Transformer pre-trained via masked autoencoder self-supervision on 1.6M unlabeled retinal images. This provides strong inductive priors for retinal structure, enabling better generalization from small datasets. Pass `--model-type retfound` to the trainer to use it.

- **Parameter-efficient fine-tuning (planned)** — LoRA adapter layers will be injected into the ViT encoder so that each interactive training step updates only a small fraction of parameters, enabling near-real-time model updates on a desktop GPU.

- **Multi-class segmentation (planned)** — A single model will simultaneously label multiple lesion types (e.g. RIPL + SDD), addressing RootPainter's one-class-per-model limitation.

- **Curriculum learning (planned)** — A staged training scheduler will present examples in order of difficulty (synthetic lesions → clear real cases → ambiguous cases → confounders), further reducing the labeled data required to reach clinical accuracy.

### Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | RETFound ViT-Large backbone + segmentation decoder | Complete |
| 2 | LoRA parameter-efficient fine-tuning | Planned |
| 3 | Curriculum learning scheduler | Planned |
| 4 | Multi-class segmentation support | Planned |

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

> **Windows users:** Python 3.12 is required (3.11 no longer provides binary installers; 3.13+ is unsupported). Use `py -3.12 -m venv env` to create the virtual environment, and activate with `env\Scripts\activate`. If activation is blocked, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` first.

> **PyTorch on Windows/Linux (CUDA):** Install PyTorch cu126 before the rest of requirements, since cu124 wheels do not exist for torch>=2.7:
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

#### Standard U-Net mode (unchanged from RootPainter)

```bash
cd trainer/src
python -u main.py --syncdir /path/to/sync_dir
```

Or, after `pip install -e .` from the `trainer/` directory:

```bash
start-trainer --syncdir /path/to/sync_dir
```

#### RETFound backbone mode

```bash
cd trainer/src
python -u main.py --syncdir /path/to/sync_dir --model-type retfound
```

Or via the entry point:

```bash
start-trainer --syncdir /path/to/sync_dir --model-type retfound
```

**First-run note:** Loading the RETFound checkpoint (~3.95 GB) takes 2–5 minutes. This is expected — the file is large and loads from disk each time the trainer starts. Once loaded, segmentation runs at roughly 3 seconds per image. Progress prints appear in the terminal during loading.

#### Painter (client)

```bash
cd painter/src/main/python
python main.py
```

When prompted, point the painter at the same sync directory used when starting the trainer.

---

### Testing

```bash
# Phase 1 — RETFound backbone (no weight download needed, uses random weights)
cd trainer/tests
python -m pytest test_retfound.py -v

# Existing unit tests (loss, unet, utilities)
python -m pytest test_loss.py test_unet.py test_utils.py -v
```

The `test_retfound.py` suite (11 tests) runs in under 15 seconds and verifies:
- ViT-Large produces the correct token shape `(B, 196, 1024)`
- `RETFoundSeg` forward pass: input `(B, 3, 224, 224)` → output `(B, 2, 224, 224)`
- Softmax probabilities sum to 1 at every pixel
- Gradients flow through the decoder

---

### References

- Zhou, K. et al. (2023). A foundation model for generalizable disease detection from retinal images. *Nature*, 622, 156–163.
- Drakopoulos, M. et al. (2024). Machine teaching allows for rapid development of automated systems for retinal lesion detection from small image datasets. *Ophthalmic Surgery, Lasers & Imaging Retina*, 55(8), 475–478.
- Smith, A.G. et al. (2022). RootPainter: deep learning segmentation of biological images with corrective annotation. *New Phytologist*, 236(2), 774–791.

---

### Contributions

This is a research fork under active development. Please open an issue or discussion before submitting a pull request.
