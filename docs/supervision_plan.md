# RetinaPainter Supervision And Uncertainty Plan

For terminology used below (sparse supervision, mask leak, `unsure`, etc.), see
[glossary.md](glossary.md).

## Current Working Decision

For now, keep **sparse corrective supervision** as the intended training policy:

- foreground correction = positive
- background correction = negative
- untouched pixels = unknown / not supervised (masked out of loss)
- future `unsure` label = explicitly ambiguous / reviewed-but-uncertain, also masked out of immediate loss

Do **not** switch to dense corrected-target training now.
Only document it as a possible future research direction.

This was confirmed by the project advisor on 2026-04-27.

## Why This Matters

RetinaPainter inherits RootPainter's corrective-annotation workflow, where the clinician corrects model mistakes instead of drawing a dense full mask for every image.

That leaves an important semantic question:

- what should an untouched pixel mean during training?

There are two broad possibilities:

1. Sparse supervision
- Only explicitly corrected pixels are supervised
- Untouched pixels are unknown and should be ignored

2. Dense corrected-target supervision
- Start from the model's previous segmentation
- Apply clinician edits
- Train on the resulting full corrected mask
- Untouched pixels effectively mean "accepted previous prediction"

For now, RetinaPainter should stay with option 1.

### Why sparse supervision wins here specifically

Densifying untouched pixels (option 2) is most useful when the model has to learn the underlying structure of the image data from scratch — every pixel of weak signal helps. RetinaPainter starts from **RETFound**, a foundation model already pre-trained on 1.6M retinal images, so the underlying-structure benefit is mostly already paid for. The remaining job is to learn the lesion-specific decision boundary, and for that, training only on what the doctors explicitly labeled is safer:

- Tiny biomarkers (RIPL, SDD) are easy to miss; treating "untouched" as background risks reinforcing missed-lesion errors and creating systematic false negatives.
- The cost of being conservative (ignoring untouched pixels) is much lower when the encoder is already pre-trained.
- For projects that did *not* have a foundation model, dense corrected-target supervision would be a more reasonable default. That distinction should be remembered if the encoder is ever swapped.

### Why `unsure` is its own annotation, not just untouched

Even though "untouched" and "unsure" are both masked out of the immediate loss, they are not the same and should not be collapsed:

1. **Protective masking with intent.** Untouched means "not reviewed." Unsure means "reviewed and the clinician judged that including this in the loss would do more harm than good" — for example, blurry / low-quality regions or genuinely ambiguous lesion edges. Recording that distinction is what makes the masking trustworthy.
2. **Per-image curriculum signal.** Images with more `unsure` pixels are, on average, harder cases. That makes `unsure` a natural difficulty score for curriculum-learning work later — pushing harder cases later in training rather than dropping them.
3. **Ground-truth uncertainty for evaluation.** Medical-imaging pipelines increasingly want a calibrated uncertainty signal. Methods exist (entropy, ensembles, abstention heads) but are hard to evaluate without something to evaluate them *against*. Clinician-marked `unsure` regions provide that "true uncertainty" benchmark.

So `unsure` is a labelling primitive that costs almost nothing in the immediate loss but unlocks several downstream uses. It also handles a real data problem: blurry or unusable image regions that the dataset already contains.

## A. Immediate Correctness And Policy-Preservation Work

### 1. Fix unlabeled-pixel loss masking

**Why:**
The current code likely intends to ignore untouched pixels, but does not exclude them properly from loss.

**Needed changes:**
- true masked CE
- true masked Dice
- true masked Tversky
- zero gradients from undefined pixels
- consistent behavior in all training paths

**Priority:** highest

### 2. Add explicit documentation of current supervision policy

**Why:**
Future contributors could otherwise accidentally change semantics while "fixing" bugs.

**Needed changes:**
- `README.md`: state current sparse supervision semantics
- `AGENTS.md`: warn that dense corrected-target training is future work only
- `CLAUDE.md`: same warning
- optionally add a short "annotation semantics" section or link to this document

### 3. Add tests specifically for supervision semantics

**Why:**
This behavior is too important to leave implicit.

**Needed changes:**
- synthetic fixtures with:
  - foreground
  - background
  - untouched
  - later `unsure`
- loss invariance test on untouched pixels
- zero-gradient test on untouched pixels
- trainer-level regression test

## B. Explicit `unsure` Annotation Support

### 4. Add explicit `unsure` label to annotation format

**Why:**
Untouched and explicitly ambiguous are not the same thing.

**Needed changes:**
- extend annotation schema / channel format
- preserve distinction between:
  - untouched
  - unsure
  - foreground
  - background
- make trainer aware of `unsure`

**Important note:**
Even if `unsure` is masked out of loss, it still has value as metadata.

### 5. Add painter UI for `unsure`

**Why:**
The label is useless without a clinician-facing tool.

**Needed changes:**
- new brush / tool
- distinct color
- keyboard shortcut
- legend / palette entry
- undo/redo/eraser support
- rendering support on load/save

### 6. Add config / compatibility support for new annotation schema

**Why:**
A third annotation state can silently corrupt old assumptions if schema compatibility is not explicit.

**Needed changes:**
- annotation schema/versioning
- legacy project compatibility
- readable failure path if migration is needed
- trainer/painter agreement on representation

### 7. Decide whether `unsure` should be excluded from metrics

**Current recommendation:** yes, by default.

**Needed changes:**
- standard segmentation metrics should ignore `unsure`
- maybe later add optional reporting:
  - percent unsure pixels
  - unsure-region counts
  - image-level uncertainty burden

## C. Data, Workflow, And UX Improvements

### 8. Add annotation-semantics decision record

**Why:**
This repo now has a nontrivial labeling-policy decision.

**Needed changes:**
- a short doc stating:
  - untouched = unknown
  - `unsure` = reviewed but ambiguous
  - dense corrected-target supervision is future work only
- include rationale for tiny retinal biomarkers

This document can serve that role if kept updated.

### 9. Update user-facing annotation instructions

**Why:**
Clinicians need to know when to use `unsure` versus leaving things untouched.

**Needed changes:**
- define:
  - when to leave untouched
  - when to mark foreground
  - when to mark background
  - when to mark unsure
- add examples for tiny lesions and ambiguous OCT regions

Without this, the label will be inconsistent across annotators.

### 10. Add project-level option or feature flag for `unsure`

**Why:**
Some studies may not want to expose it immediately.

**Possible change:**
- project setting:
  - `enable_unsure_annotations: true/false`

This is optional, but worth considering if mixed workflows are expected.

## D. Research And Experimental Work

### 11. Evaluate sparse supervision versus dense corrected-target supervision

**Why:**
This is still an open research question for RIPL and SDD.

**Experimental arms:**
- sparse supervision
- dense corrected-target supervision
- staged or hybrid supervision

**Metrics:**
- lesion-level recall
- lesion-level precision
- false-positive burden
- Dice / IoU
- learning speed
- robustness to missed lesions

This should stay separate from bug-fix work.

### 12. Consider a staged or hybrid supervision policy later

**Possible future policy:**
- early training: untouched = unknown
- later training on strong models: optionally accept untouched predictions in reviewed images

This is not for now, but it should be recorded as a possible future path.

### 13. Use `unsure` labels for curriculum learning later

Possible later uses:
- image difficulty score
- sampling harder cases
- curriculum stage transitions
- disagreement-based review queue

This should be captured now so the `unsure` label is not treated as dead metadata.

### 14. Use `unsure` labels for uncertainty evaluation later

Possible later use:
- compare model uncertainty maps against clinician-marked unsure regions
- evaluate "true uncertainty" rather than generic entropy only

Again: not immediate, but worth tracking.

## E. Future UI And Model Work

### 15. Add model-predicted uncertainty overlay later

**Not now.**

Possible later work:
- entropy overlay
- ensemble disagreement overlay
- abstention / uncertainty head
- comparison to clinician `unsure`

This should remain a separate long-term task.

### 16. Add export and reporting support for `unsure`

Optional but useful later:
- export `unsure` masks
- include uncertainty burden in per-image CSVs
- enable retrospective audit of ambiguous regions

Not immediate, but worth tracking.

## F. Additional Items That Should Be Added To The Plan

### 17. Backward-compatibility and migration note

If annotation format changes, explicitly decide:
- will old projects auto-upgrade?
- or remain two-state only?
- or require migration?

This should be in the plan, not discovered during implementation.

### 18. Metrics and plots impact audit

If `unsure` exists, check:
- training metrics
- validation metrics
- segmentation metrics plot
- CSV export logic
- any "corrected segmentation" logic

This is easy to miss.

### 19. Corrected-segmentation logic audit

Some parts of the painter already compute "corrected segmentation" by starting from the model segmentation and applying corrections for plotting metrics.

That logic may need explicit review once `unsure` exists, because:
- should unsure overwrite anything?
- should it leave the prediction unchanged?
- should it mask the region from evaluation?

This deserves its own review even if not a full ticket.

### 20. Annotation ergonomics and color design review

If `unsure` is added:
- pick a color distinct from foreground and background
- ensure it is visible on OCT grayscale imagery
- ensure it does not get confused with segmentation overlay colors

This sounds small, but it matters.

## Recommended Implementation Order

Each phase below lists the **specific code changes** required, anchored to current files. File paths use the layout in this repo as of 2026-04-27.

### Phase 1 — Fix sparse supervision so it actually works

Goal: untouched pixels contribute zero gradient and zero loss. No new annotation category yet.

1. **Document the current policy in code-adjacent locations.**
   - `README.md`, `AGENTS.md`, `CLAUDE.md` already note the policy at a high level — extend with a short "Annotation semantics" subsection that links to this doc.
   - Add a short docstring at the top of `trainer/src/loss.py` and `trainer/src/datasets.py` stating that `mask` / `defined` represents "supervised pixels only" and that untouched pixels must not contribute to loss.

2. **Fix the leakage in `trainer/src/trainer.py:374–379`.**
   - Currently:
     ```python
     outputs[:, 0] *= defined_tiles
     outputs[:, 1] *= defined_tiles
     loss = combined_loss(...) or tversky_loss(...)
     ```
     Zeroing logits is **not** ignoring those pixels — softmax(0,0) = (0.5, 0.5), so each undefined pixel still contributes a constant CE penalty of `log 2`, and Dice/Tversky still see them in numerator and denominator.
   - Change: pass `defined_tiles` through to the loss functions and apply the mask **inside the loss** (after softmax, before reduction).
   - Remove the `outputs[:, c] *= defined_tiles` lines — they are no longer needed once losses honour the mask.

3. **Add masked variants in `trainer/src/loss.py`.**
   - `combined_loss(predictions, labels, mask)` → masked CE (`reduction='none'` → multiply by mask → `sum / mask.sum()`) + masked Dice (intersection and union restricted to `mask > 0`).
   - `tversky_loss(predictions, labels, mask, ...)` → restrict `tp`, `fn`, `fp` sums to `mask > 0`. Keep the per-class loop and class weights.
   - Guard the `mask.sum() == 0` case (all-undefined tile) — return `0` as a tensor that still requires grad rather than dividing by zero.

4. **Verify `outputs.mul_(defined_tiles)` is actually gone from the metrics path too.**
   - Lines 382 (`foreground_probs *= defined_tiles`) and 390–399 already correctly restrict metrics to defined pixels. Keep that path; it is the model the loss should follow.

5. **Fix the latent overlap bug in `trainer/src/datasets.py:161`.**
   - Current: `mask = foreground + background`. This is arithmetic addition of `{0,1}` channels — equal to logical OR **only if** `foreground` and `background` are mutually exclusive at every pixel. If a malformed annotation marks a pixel as both, `mask = 2` there, which silently double-weights that pixel in any masked loss and (under current code) doubles its logits via `outputs *= defined_tiles`.
   - Change: assert no overlap, then use logical OR.
     ```python
     assert not np.any(foreground & background), \
         f"Annotation has overlapping fg/bg pixels: {fname}"
     mask = np.logical_or(foreground, background).astype(np.float32)
     ```
   - The assertion catches painter / data-pipeline bugs; the OR makes the {0,1} contract obvious to readers.

5. **Add regression tests in `trainer/tests/test_loss_masking.py` (new).**
   - Synthetic `(B, 2, H, W)` logits + label tile with three regions: foreground (label=1, mask=1), background (label=0, mask=1), untouched (label=0, mask=0).
   - Assert: changing logits inside the untouched region does **not** change loss value (loss invariance).
   - Assert: gradient w.r.t. logits is exactly zero in the untouched region.
   - Assert: a fully-defined tile produces the same loss as the legacy `combined_loss` did before the change (within tolerance), so behavior on already-supervised pixels does not regress.
   - Same three tests for `tversky_loss` with the `retfound_rfa` defaults.

6. **Smoke-train both `--model-type retfound` and `--model-type unet` on an existing project** to confirm loss curves still descend and the loss values are now lower (because the `log 2` constant penalty is gone). Document expected ~0.7 nat shift in commit message so future-Donna does not interpret it as a regression.

### Phase 2 — Add the `unsure` annotation primitive

Goal: clinicians can mark a third state, the trainer knows to mask it, and old projects keep working.

1. **Define the on-disk schema.**
   - Annotations are saved as RGBA PNGs; trainer reads `annot[:, :, 0]` as foreground, `annot[:, :, 1]` as background, channel 2 is currently unused (`trainer/src/datasets.py:118` already takes `[:, :, :2]`). Use **channel 2 as `unsure`**.
   - This is naturally backward-compatible: legacy 2-channel projects load with channel 2 all zero, which means "no unsure pixels" — same as current behaviour.
   - Add a `schema_version` field somewhere project-level (project JSON or a small `annotation_schema.json`). Bump from implicit v1 (foreground+background) to v2 (foreground+background+unsure).

2. **Painter — add the `Unsure` brush.**
   - `painter/src/main/python/palette.py:114` — extend `default_brushes` to include `('Unsure', (255, 200, 0, 180), '3')` (yellow/amber, distinct from red foreground and green background; visible on grayscale OCT). Keep `Background` hardcoded in `get_brush_data` and add `Unsure` next to it as a non-removable default.
   - Wire the brush to channel 2 in the save path (`painter/src/main/python/file_utils.py:93+` `maybe_save_annotation`). Confirm the QPixmap → PNG round-trip preserves channel 2; if the paint surface is RGB-only, switch to RGBA paint.
   - Update `convert_seg.py`, `assign_corrections.py`, and the load/render path in `im_viewer.py` / `graphics_scene.py` to round-trip channel 2.
   - Keyboard shortcut, eraser, undo/redo, legend entry — same plumbing as foreground/background, just one more channel.

3. **Trainer — extend dataset and loss masking.**
   - `trainer/src/datasets.py:118` — change to `annot = annot[:, :, :3]`. Split out an `unsure = annot[:, :, 2]` channel.
   - Redefine the supervision mask correctly:
     ```python
     # foreground, background, unsure must be mutually exclusive
     assert not np.any(foreground & background), f"fg/bg overlap: {fname}"
     assert not np.any(foreground & unsure),     f"fg/unsure overlap: {fname}"
     assert not np.any(background & unsure),     f"bg/unsure overlap: {fname}"
     defined = np.logical_or(foreground, background)
     mask    = (defined & ~unsure.astype(bool)).astype(np.float32)
     ```
     This guarantees `mask ∈ {0, 1}` and makes the "unsure pixels are excluded from loss" semantic explicit. Use the same pattern in Phase 1 step 5 (logical OR + assertion) — do not regress to arithmetic addition.
   - Return both `mask` and `unsure` from `__getitem__` so the trainer can log "% unsure" without re-loading the annotation.
   - Confirm `tile_pad > 0` cropping still applies symmetrically to the unsure channel.

4. **Trainer — metrics and CSVs.**
   - `trainer/src/trainer.py:415` `log_metrics` — exclude unsure pixels from precision/recall/F1 (already follows the `defined` mask, so this falls out automatically once the mask is correct).
   - Add a new column `unsure_frac` to the train/val CSVs: `unsure.sum() / unsure.numel()` averaged across the epoch. Header at line 427–428 needs updating.
   - `painter/src/main/python/plot_seg_metrics.py` — make sure the plotting path tolerates the new column (additive, so old runs without the column should still render).

5. **Painter — corrected-segmentation logic audit.**
   - There are places in the painter that compute "what the corrected segmentation would look like" by overlaying user corrections on the current model output (search `corrected` / `assign_corrections.py`). Decide explicitly: an unsure pixel **leaves the prediction unchanged for visualisation but excludes the pixel from any quantitative metric**. Document this in the file's header comment.

6. **Backward compatibility.**
   - On project load, if `schema_version` is missing, treat as v1 → channel 2 = zero, and silently upgrade in memory. Do **not** rewrite old annotation files on disk unless the user opts in.
   - Add a one-line warning in the painter status bar when an unmigrated v1 project is opened ("This project does not use the Unsure label. Saving will keep it 2-channel.") — or always upgrade on save. Pick one and stick to it.

7. **User-facing instructions.**
   - Add a short "When to use Unsure" section to the painter's help dialog or a new page in `docs/`. Anchors:
     - Use foreground for definite lesion, background for definite non-lesion.
     - Use unsure for: blurry / unusable regions, lesion edges where you cannot commit, image artefacts, anatomy you have not been asked to label.
     - Leave untouched for everything else (the model will not be told anything about those pixels either way).

8. **Tests.**
   - Extend `test_loss_masking.py` with a fourth region: `unsure` (channel 2 = 1, foreground = 0, background = 0). Same loss-invariance and zero-gradient assertions.
   - Add a `test_annotation_schema.py` round-trip test: save a 3-channel PNG, load it through `datasets.py`, assert mask logic produces the right `defined` and `unsure` tensors.
   - Add a backward-compatibility test: load a 2-channel PNG and confirm channel 2 reads as all zeros.

9. **Optional project flag `enable_unsure_annotations`.** Skip until a study explicitly asks for it — feature-flag drift is its own cost.

### Phase 3 — Research and curriculum experiments

These are research deliverables, not code-cleanup. Each one needs an explicit experimental design and a write-up before merging anything beyond infrastructure to support it.

1. **Sparse vs. dense corrected-target supervision (paper-quality comparison).**
   - Implement a `--supervision-mode` flag (`sparse` default, `dense_corrected` opt-in). In `dense_corrected` mode, on each training step build the label tile by starting from the previous segmentation and applying corrections.
   - Run on RIPL and SDD datasets; report Dice, lesion-level recall/precision, false-positive burden, learning-speed curves.
   - Pre-register the hypothesis ("foundation-model + sparse beats foundation-model + dense on small lesions") so the result is interpretable either way.

2. **Curriculum learning driven by `unsure` density.**
   - Per-image difficulty score = `unsure_pixels / total_pixels` (or a smarter aggregate).
   - Sampling strategy: easy → hard within an epoch, or staged epochs.
   - Compare against random sampling. Use `unsure_frac` already logged in Phase 2.

3. **Hybrid / staged supervision.**
   - Start with sparse, and after the model crosses a confidence threshold per image, allow accepting untouched-but-confidently-predicted pixels into the loss.
   - This is a research experiment, not a default. Keep it gated.

### Phase 4 — Uncertainty surfacing and exports

Long-tail UI / reporting work. None of this is required for the labelling-policy fix, but it is what `unsure` annotations enable.

1. **Model-predicted uncertainty overlay.** Entropy or ensemble disagreement, rendered in the painter alongside the segmentation. Compare against clinician `unsure` regions.
2. **Uncertainty-aware metrics.** Calibration plots, Brier score, abstention quality — evaluated on held-out images that have `unsure` annotations.
3. **Exports.** Add `unsure` masks to per-image CSVs and any segmentation export. Useful for retrospective audit and downstream analysis.

## Summary

The current safest path is:

- keep sparse corrective supervision
- fix the implementation so untouched pixels are truly ignored in loss
- add an explicit `unsure` label later as a separate semantic category
- record dense corrected-target training as future research only

The main thing to avoid is silently drifting from "untouched = unknown" to "untouched = accepted prediction" during bug-fix work.
