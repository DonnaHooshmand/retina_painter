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

#### Trainer (server)

```bash
cd trainer
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

New dependencies added for the RETFound backbone: `timm>=0.9.0` and `huggingface_hub>=0.20.0` (both included in `requirements.txt`). RETFound weights (~330 MB) are downloaded automatically from HuggingFace Hub on first use and cached at `~/.cache/retina_painter/`.

#### Painter (client)

```bash
cd painter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

### Running

#### Standard U-Net mode (unchanged from RootPainter)

```bash
cd trainer/src && python main.py --syncdir ~/root_painter_sync
```

Or, after pip install:

```bash
start-trainer --syncdir ~/root_painter_sync
```

#### RETFound backbone mode

```bash
start-trainer --syncdir ~/root_painter_sync --model-type retfound
```

On first run this downloads RETFound OCT weights (~330 MB) from HuggingFace Hub. Subsequent runs use the cached file. The trainer will use 224×224 patches (instead of 572×572) and a conservative batch size to accommodate the larger model.

If you are on a machine without internet access, download `RETFound_oct.pth` manually from [rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE) and place it at `~/.cache/retina_painter/RETFound_oct.pth`.

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
