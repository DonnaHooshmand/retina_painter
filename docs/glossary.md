# RetinaPainter Glossary

Working vocabulary for RetinaPainter, with the paper in mind. Each entry locks in
the meaning we use *in this project* — generic ML terms are only listed when our
usage is narrower or more loaded than the textbook one. Update this file whenever
a term takes on a new shade of meaning during research; do not silently redefine.

## A. Supervision and annotation

### Corrective annotation
The interaction model inherited from RootPainter. The clinician does not draw a
dense ground-truth mask. Instead, the model proposes a segmentation, and the
clinician paints over pixels the model got wrong. The training signal is
therefore concentrated on errors, not on every pixel of every image.

### Sparse supervision (a.k.a. sparse corrective supervision)
The training policy where **only pixels the clinician explicitly painted** count
toward the loss. Untouched pixels are treated as unknown and contribute zero
gradient and zero loss. This is RetinaPainter's chosen policy as of 2026-04-27.
Contrast with *dense corrected-target supervision*.

### Dense corrected-target supervision
An alternative training policy where the label tile is built by taking the
model's previous segmentation and overlaying the clinician's corrections, so
every pixel has a label and contributes to the loss. Untouched pixels are
implicitly treated as "the model was right." Recorded as future research only;
not the current contract.

### Foreground / background / untouched / unsure
The four annotation states a pixel can be in:
- **foreground** — clinician marked this pixel as lesion / target class.
- **background** — clinician marked this pixel as definitely not lesion.
- **untouched** — clinician did not paint here. Means *unknown / not reviewed*.
  Masked out of loss.
- **unsure** — clinician reviewed this pixel and judged it ambiguous (blurry,
  artefact, uncommittable lesion edge). Also masked out of loss, but **not** the
  same as untouched: it is reviewed-and-deferred, which gives it metadata value
  for curriculum work and uncertainty evaluation. See
  [supervision_plan.md](supervision_plan.md).

### Supervision mask (a.k.a. `defined`, `mask`)
The binary tensor passed to the loss that says which pixels are supervised
(`1`) vs. ignored (`0`). In the current implementation, `mask = (foreground OR
background) AND NOT unsure`. The contract is: a pixel with `mask == 0` must
contribute zero loss-value sensitivity and zero gradient.

### Mask leak (the bug fixed in commit `08cba0e`)
Project-specific term for the failure mode where untouched pixels appeared to
be ignored but were not. Old code did `outputs[:, c] *= defined_tiles`, which
zeroed the logits but left `softmax(0, 0) = (0.5, 0.5)`, so each untouched
pixel still incurred a constant `~log 2` CE penalty plus a Dice/Tversky
contribution. "True masking" means applying the mask *inside* the loss after
softmax.

### Annotation router
The painter-side logic that decides whether a new annotation goes to
`annotations/train/` or `annotations/val/`. The default router maintains a 5:1
file-count ratio with no awareness of patient ID; the `train-val-split` branch
adds an explicit-folder mode.

## B. Data-splitting hygiene

### Patient-level data leakage
The failure mode where the same patient's images appear in both training and
validation sets. For retinal OCT this is severe because scans from one patient
are nearly identical, so val-F1 inflates and early stopping fires too late.
Affects only the **internal validation signal**; the held-out test set is
always a separate physical folder.

### Internal validation set
The split the trainer uses for model selection and early stopping
(`<project>/annotations/val/`). Its integrity depends on the annotation
router; this is what `train-val-split` protects.

### Held-out test set
A physical folder *outside* the painter project, used only for final reporting.
Never touched by the annotation router, so its integrity is preserved by
construction.

## C. Model architecture

### Foundation model
A model pre-trained on a large corpus and reused as a frozen (or partially
frozen) feature extractor. In RetinaPainter, the foundation model is RETFound.

### RETFound
The ViT-Large backbone pre-trained by Zhou et al. on 1.6M retinal images.
Loaded from the MAE checkpoint `RETFound_oct.pth` on Hugging Face
(`monish563/RETFOUND`) or via Google Drive through `setup_retfound.py`.
(`iszt/RETFound_mae_natureOCT` is a separate Transformers `model.safetensors`
release.) Used as the encoder in both `retfound` and
`retfound_rfa` model types. ViT-Large = patch_size 16, embed_dim 1024,
depth 24, num_heads 16.

### `retfound` (plain decoder)
RETFound encoder + a 4-stage `ConvTranspose2d` upsampler that maps the
`(B, 196, 1024)` patch tokens to a `(B, 2, 224, 224)` segmentation. No skip
connections. Trained with combined loss + SGD.

### `retfound_rfa` (RFA-U-Net decoder)
Same RETFound encoder, but with a U-Net-style decoder that pulls intermediate
ViT block outputs (Z6, Z12, Z18, Z24) as skip connections, gated by additive
attention before concatenation. Implements Hayati et al. (2025). Trained with
Tversky loss + AdamW, with the first 21 of 24 ViT blocks frozen.

### Attention gate (additive attention)
The mechanism the RFA decoder uses to filter skip-connection features:
`Wg + Ws → ReLU → 1×1 conv → sigmoid → scale` applied to the skip tensor.
"Additive" because the gate sums the two projections rather than dot-producting
them. Distinct from the multi-head attention inside the ViT encoder.

### Skip pyramid
The progressive `ConvTranspose2d` chain that brings a `(14, 14)` ViT feature
map up to the resolution of a given decoder stage (28, 56, or 112) while
reducing channels. Implemented as `_make_skip_pyramid` in
[retfound_rfa_model.py](../trainer/src/retfound_rfa_model.py).

### Encoder freezing
Setting `requires_grad = False` on the first N transformer blocks so only the
last few blocks plus the decoder train. We freeze the first 21 of 24 blocks
for `retfound_rfa`, matching the RFA-U-Net paper. Different from "the encoder
is fully frozen" — three blocks remain trainable.

## D. Loss and optimization

### Combined loss
`0.7 · Dice + 0.3 · CrossEntropy`. Default loss for `unet` and `retfound` model
types. Implemented in [loss.py](../trainer/src/loss.py); accepts an optional
`mask` argument (post-mask-leak fix).

### Tversky loss
Generalized Dice with separate weights for false positives (`alpha`) and false
negatives (`beta`). We use `alpha=0.7, beta=0.3` plus `class_weights=(1.0,
2.0)` to up-weight the foreground class. Default loss for `retfound_rfa`.
Better suited to small / rare foreground regions like RIPL and SDD, where
false negatives are the costlier error mode.

### Optimizer choice (project convention)
`unet` and `retfound` use SGD; `retfound_rfa` uses AdamW (lr 1e-4, weight
decay 1e-4) on trainable params only. Recorded here because reviewers will ask.

## E. Tiling and inference

### Tile / patch
A fixed-size crop of an image fed to the model. UNet uses 572×572 input → 500×500
output (valid convolutions). RETFound uses 224×224 input → 224×224 output.

### `tile_pad`
The asymmetry between input and output crop sizes, equal to `(in_w - out_w) //
2`. For UNet `tile_pad = 36`; for both RETFound variants `tile_pad = 0`. Any
code that crops by `tile_pad` must guard with `if tile_pad > 0:` because
`x[0:-0]` in Python is `x[0:0]` — empty, not a no-op.

## F. Evaluation and reporting

### Lesion-level recall / precision
Recall and precision computed at the level of *connected lesions* rather than
individual pixels: a predicted blob counts as a true positive if it overlaps a
ground-truth lesion above some threshold, regardless of pixel-level Dice.
Important for tiny-biomarker work (RIPL, SDD), where pixel-Dice is dominated
by background.

### False-positive burden
Number of spurious predicted lesions per image (or per patient). Pairs with
lesion-level recall as a reporting metric for clinical usefulness.

### Unsure fraction (`unsure_frac`)
Per-image or per-epoch fraction of pixels marked `unsure`. Logged in the
training CSV. Doubles as a per-image difficulty score for curriculum work.

## G. Roadmap terminology

### Phase 1 / Phase 2 / Phase 3 / Phase 4
The roadmap stages used throughout `CLAUDE.md` and `supervision_plan.md`:
1. **Phase 1** — RETFound backbone integration *and* the sparse-supervision
   mask-leak fix. Done.
2. **Phase 2** — Explicit `unsure` annotation primitive, schema versioning,
   painter UI. In progress on `unsure-annotation`.
3. **Phase 3** — Research experiments: sparse vs. dense supervision
   comparison, curriculum learning driven by `unsure` density, hybrid /
   staged supervision.
4. **Phase 4** — Uncertainty surfacing in the UI, calibrated metrics, exports.

### LoRA fine-tuning
Low-Rank Adaptation: inject small trainable rank-decomposed matrices into the
frozen attention blocks so only those plus the decoder train. Listed under
Phase 2 in `CLAUDE.md`'s development status; not yet implemented.

### Curriculum learning (this project's flavor)
Ordering training images by difficulty rather than uniformly random. The
difficulty signal we plan to use is `unsure_frac` — clinician-marked
uncertainty as a proxy for case hardness. Listed under Phase 3.
