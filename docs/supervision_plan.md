# RetinaPainter Supervision And Uncertainty Plan

## Current Working Decision

For now, keep **sparse corrective supervision** as the intended training policy:

- foreground correction = positive
- background correction = negative
- untouched pixels = unknown / not supervised
- future `unsure` label = explicitly ambiguous / reviewed-but-uncertain, also masked out of immediate loss

Do **not** switch to dense corrected-target training now.
Only document it as a possible future research direction.

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

### Phase 1
1. Document current sparse supervision semantics
2. Fix unlabeled-pixel masking in loss
3. Add tests for untouched-pixel exclusion

### Phase 2
4. Add explicit `unsure` label to annotation format
5. Add painter UI for `unsure`
6. Add schema/version/backward-compat support
7. Audit metrics and corrected-segmentation logic
8. Update user instructions for annotators

### Phase 3
9. Run sparse versus dense corrected-target supervision experiment
10. Explore curriculum use of `unsure`

### Phase 4
11. Model-predicted uncertainty overlay
12. Uncertainty evaluation tooling and exports

## Summary

The current safest path is:

- keep sparse corrective supervision
- fix the implementation so untouched pixels are truly ignored in loss
- add an explicit `unsure` label later as a separate semantic category
- record dense corrected-target training as future research only

The main thing to avoid is silently drifting from "untouched = unknown" to "untouched = accepted prediction" during bug-fix work.
