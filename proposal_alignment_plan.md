# RetinaPainter — Proposal Alignment Plan (things to do & how)

**Date:** 2026-06-11
**Source of truth:** `retinapainter- Proposal.pdf` (Sections 1.5 Proposed Work, 1.6 Timeline, Fig 1.4)
**Companion doc:** `code_review_findings.txt` (the pipeline quirks/bugs referenced below by `[CRF #n]`)

This file maps **what the proposal promises** onto **what the code does today**, then lists
the concrete work to close the gap — with the specific files, functions, and steps for each.
It is ordered so that the cheap correctness fixes (which the proposal's data-efficiency
claims silently depend on) come before the big new features.

---

## 1. The proposal's functional contract

The proposal (1.5) commits RetinaPainter to **four** pillars. Everything below is judged
against these:

| # | Pillar (proposal 1.5) | One-line commitment |
|---|---|---|
| P1 | **Foundation-model backbone** | Replace U-Net-from-scratch with the RETFound ViT, pre-loaded with retinal weights. |
| P2 | **Multi-class segmentation** | One model labels multiple lesion types at once (RIPL + SDD + Drusen in Fig 1.4) — fixes RootPainter's one-class-per-model limit. |
| P3 | **Real-time interactive loop via parameter-efficient tuning** | LoRA / adapter layers so each interactive training step updates only a tiny parameter subset → "near real-time updates on a desktop GPU." |
| P4 | **Curriculum learning module** | Automated 4-stage schedule: (1) synthetic lesions → (2) easy real → (3) hard real → (4) confounders; user tags images easy/hard; heavy augmentation early. |
| P5 | **Open-science evaluation** | Released forked repo + example data + pre-trained models, with a rigorous eval on RIPL/SDD: *annotation-time-to-accuracy*, held-out test performance, user feedback. |

(The proposal also promises the interactive corrective-annotation GUI from RootPainter — that
part exists and is unchanged.)

---

## 2. Status matrix (proposal → code today)

| Pillar | Status | Where it lives / what's missing |
|---|---|---|
| P1 Foundation backbone | **DONE** | `model_utils._build_model`, `retfound_vit.py`, `retfound_model.py`, `retfound_rfa_model.py`; selectable via `--model-type retfound\|retfound_rfa`. |
| P2 Multi-class | **NOT STARTED** | Everything is hard-wired to `num_classes=2` and a binary fg/bg annotation encoding. Needs decoder head + loss + metrics + **painter** changes. |
| P3 LoRA / real-time | **NOT STARTED** | No LoRA/adapter/peft anywhere. Today the whole encoder (or last 3 blocks for `retfound_rfa`) is fine-tuned with full-precision SGD/AdamW. |
| P4 Curriculum | **NOT STARTED** | Training samples tiles *uniformly at random* (`im_utils.load_train_image_and_annot`); no notion of stage, difficulty, or synthetic data. |
| P5 Evaluation | **PARTIAL** | Per-epoch train/val CSVs exist (`trainer.log_metrics`); no held-out test scorer, no annotation-time-to-accuracy curve, no per-class report. |

---

## 3. Prerequisite fixes (do these first — the proposal's claims depend on them)

These are small, but each one directly undermines a headline proposal claim if left as-is.
Full detail is in `code_review_findings.txt`; here is *why each blocks the proposal* and the fix.

### 3.1 NaN-F1 vs early stopping `[CRF #6]` — **blocks "reliable detection from 100–200 images"**
- **Problem:** when the model predicts no foreground (normal early on for rare RIPL/SDD), val F1 is `NaN`, which counts as "no progress." After 60 such epochs training auto-stops — possibly *before the model ever learns the lesion*. This makes the central data-efficiency claim unreliable and corrupts the P5 time-to-accuracy metric.
- **How:** in `trainer/src/metrics.py:47-53`, return `f1 = 0.0` (not `NaN`) when `tp == 0`; or have `save_if_better`/early-stop fall back to IoU or recall while F1 is degenerate. Make `max_epochs_without_progress` (`trainer.py:132`) a CLI arg.
- **Done when:** a from-scratch RIPL run no longer stops while still predicting all-background.

### 3.2 Per-epoch prev-model reload `[CRF #5]` — **blocks "near real-time updates" (P3)**
- **Problem:** `Trainer.validation()` (`trainer.py:461-464`) re-reads the previous checkpoint from disk **every epoch** and `deepcopy`s the live model. For RETFound that's re-loading ~1.2 GB per epoch — incompatible with "responsive desktop-GPU" interactivity.
- **How:** cache the previous-best model in memory (`self.prev_model`, `self.prev_f1`); only reload when a new best is saved. Run validation on `self.model` under `eval()/no_grad()` instead of `deepcopy`. (LoRA in 5.2 also shrinks checkpoints, compounding the win.)
- **Done when:** a validation epoch does zero disk model-loads in steady state.

### 3.3 Sampling coverage + per-image weighting `[CRF #2, #3]` — **foundation for P4 and for the efficiency claim**
- **Problem:** tiles are drawn uniformly with replacement; an image with one tiny lesion is sampled as often as a richly-annotated one, and beyond ~306 train images some are unseen each epoch. The curriculum module (P4) *is* a smarter sampler, so fixing this is the natural first step toward it.
- **How:** replace the uniform `random.sample` in `im_utils.load_train_image_and_annot:71` with a weighted/round-robin sampler in `datasets.TrainDataset` (weight by annotated-foreground pixel count or lesion-component count). Add a max-attempts guard to the `while True` tile-acceptance loop (`datasets.py:136-142`).
- **Done when:** every annotated image is guaranteed exposure per epoch, and lesion-dense images are oversampled.

### 3.4 Honest train/val split `[CRF #4]` — **blocks honest P5 evaluation**
- **Problem:** the painter routes annotations train/val by a 5:1 *count* ratio with no patient grouping, leaking near-duplicate B-scans across the split and inflating val-F1 / early-stopping. The proposal's whole selling point is measured accuracy, so the split must be patient-clean.
- **How:** finish the planned `train-val-split` branch (explicit train/val source folders chosen at project creation) so routing is by source folder, not count. Until then, standardize on `prepare_annotations.py` pre-population.
- **Done when:** no patient appears in both `annotations/train/` and `annotations/val/`.

### 3.5 Quick correctness `[CRF #8, #9, #10]`
- `trainer.py:339` photo guard should be `any(is_photo(a) ...)`; `metrics.py:41` needs a `total == 0` guard; add a `worker_init_fn` to the DataLoader (`trainer.py:352`) to re-seed NumPy per worker so Gaussian/salt-pepper augmentation isn't duplicated across workers.

---

## 4. P2 — Multi-class segmentation

**Goal:** one model outputs N lesion classes (RIPL, SDD, Drusen, …) + background, per Fig 1.4.

**Current state:** `num_classes=2` everywhere; annotations are binary (R channel = foreground,
G channel = background); loss/metrics assume one foreground class.

**This is the largest change because it crosses the painter↔trainer boundary** (CLAUDE.md notes
the painter is otherwise unchanged). Recommended order:

### 4.1 Decide the annotation encoding (do this first — everything else follows)
- Today: RGBA PNG, `annot[:,:,0]`=fg, `annot[:,:,1]`=bg, "defined" = fg∨bg.
- **Recommended:** a **2-channel label PNG** — channel 0 = class index `0..N` (0 = background), channel 1 = "defined" mask (1 where the clinician painted anything). This scales past 3 classes (RGB-channel-per-class does not) and keeps the sparse-supervision contract intact.
- Document it in `docs/` and version it (e.g. a `label_format: 2` field in the project file) so old binary projects still load.

### 4.2 Trainer side
- **Model:** plumb `num_classes` from a new `--num-classes` CLI arg through `Trainer` → `_build_model` → `RETFoundSeg`/`RETFoundSegRFA`. The decoder head is already `Conv2d(64, num_classes, 1)` (`retfound_model.py:120`), so the architecture change is trivial; the work is the data path.
- **Loss (`loss.py`):** generalize `combined_loss`/`tversky_loss` from "softmax channel 1" to multi-class — per-class Dice/Tversky averaged over classes + masked cross-entropy over all classes. Keep the `mask` (defined) semantics exactly as today.
- **Dataset (`datasets.py`):** stop splitting fg/bg into two channels; load the class-index map + defined mask. Update the augmentation path (elastic/flip must carry the integer label map without interpolating across class boundaries — use nearest-neighbor for labels).
- **Metrics (`metrics.py`, `model_utils.get_val_metrics`):** compute per-class TP/FP/FN and report per-class + macro F1/IoU. Update the CSV header in `trainer.log_metrics`.

### 4.3 Painter side (the part RootPainter never had)
- Add a **class palette / brush selector** (N colors) and write the chosen class into the new label encoding when saving (`maybe_save_annotation` path in `painter/.../file_utils.py`).
- Multi-class overlay rendering of predictions (distinct colors per class).
- The train/segment **instruction JSON** gains `num_classes` / class names; keep backward-compatible defaults so binary projects still work.

**Done when:** a project annotated with RIPL+SDD trains one model whose validation CSV shows a
non-trivial per-class F1 for both, and predictions render in two colors.

---

## 5. P3 — LoRA / parameter-efficient fine-tuning (real-time loop)

**Goal:** each interactive step updates only adapter + decoder params → fast updates on a
desktop GPU (proposal 1.5, P3).

**Current state:** `retfound` fine-tunes the whole ViT with SGD; `retfound_rfa` already freezes
21/24 blocks (`trainer.start_training:300-308`) but still full-tunes the last 3 + decoder. No LoRA.

**How:**
1. **Adapter layers:** add LoRA to the ViT attention projections in `retfound_vit.py` (the `qkv` and output `proj` Linear layers in each block). Either use the `peft` library or a small hand-rolled `LoRALinear` (rank `r`, scaling `alpha`) wrapping the existing Linear. Add a `lora_rank` constructor arg.
2. **New model type:** register `retfound_lora` (and/or a `--lora-rank` flag usable with the existing RETFound variants) in `model_utils._build_model`, `main.py`, and `src/__init__.py` (keep those two in sync — CLAUDE.md). Respect the existing rule that `retfound`/`retfound_rfa` checkpoints are mutually incompatible — LoRA is a third, separate checkpoint family.
3. **Optimizer:** in `Trainer.start_training`, freeze the base encoder; build `AdamW` over **only** `requires_grad` params (LoRA + decoder), as already done for `retfound_rfa`.
4. **Checkpoints:** save **only** the LoRA + decoder state (a few MB) rather than the full ViT. This shrinks the `.pkl`, which *also* fixes the per-epoch reload cost in 3.2 and makes interactive saving snappy.
5. **Benchmark:** measure wall-clock per training step (LoRA vs full fine-tune) to substantiate the "near real-time" claim — feed this into P5.

**Done when:** an interactive correction triggers a training step that is meaningfully faster
than full fine-tuning, and the saved checkpoint is small.

---

## 6. P4 — Curriculum learning module

**Goal:** automated staged training — synthetic → easy → hard → confounders — with user-supplied
easy/hard tags and heavy early augmentation (proposal 1.5, P4 + Fig 1.4).

**Current state:** none; uniform random sampling (see 3.3, which is the prerequisite refactor).

**How:**
1. **Difficulty source:** a per-project `curriculum.json` mapping each image filename → `{stage|difficulty}` (`synthetic`/`easy`/`hard`/`confounder`). Start hand-authored; later add a painter affordance to tag the current image easy/hard (small UI addition, writes into `curriculum.json`).
2. **Stage scheduler (`trainer/src/curriculum.py`, new):** holds the ordered stages and the "active pool" of filenames. The `TrainDataset` sampler (from 3.3) draws only/mostly from the active pool.
3. **Stage advancement:** advance when val-F1 plateaus (reuse `epochs_without_progress`) or on a fixed schedule; optionally keep a fraction of earlier-stage data to avoid forgetting.
4. **Synthetic-lesion stage (Stage 1):** a data-prep script (e.g. `scripts/make_synthetic_lesions.py`) that pastes synthetic RIPL/SDD-like blobs into normal OCTs and emits images + label PNGs in the new multi-class format, so Stage 1 is "just another dataset." This is the most research-heavy piece — scope it as a standalone deliverable.
5. **Augmentation-by-stage:** parametrize `UNetTransformer` (`datasets.py:64`) with a strength knob; apply heavy augmentation in early stages, lighter later.

**Done when:** a training run visibly progresses through stages (logged), starting on
synthetic/easy data and ending on hard/confounder cases, with stage transitions in the logs.

---

## 7. P5 — Evaluation & open-science harness

**Goal:** reproducible evidence for the proposal's metrics: annotation-time-to-accuracy,
held-out test performance, usability (proposal 1.5, P5).

**Current state:** per-epoch train/val CSVs only; no test scorer; no time-to-accuracy.

**How:**
1. **Held-out test scorer (`scripts/evaluate.py`, new):** segment a physically separate test folder (already kept outside the project per CLAUDE.md) and compute per-class precision/recall/F1/IoU against ground-truth masks, reusing `metrics.get_metrics`. Output a CSV + summary.
2. **Annotation-time-to-accuracy (`scripts/annotation_efficiency.py`, new):** model checkpoints already embed a UNIX timestamp in their filename (`model_utils.save_if_better`), and the painter records interaction time (`painter/.../interaction_time.py`). Join them: for each checkpoint, plot test-F1 vs cumulative annotation minutes → the proposal's headline curve (mirrors Fig 1.3's "labeled pixels over time", but for accuracy).
3. **Per-class reporting:** once P2 lands, extend the train/val CSV headers and the painter's `plot_seg_metrics.py` to show per-class curves.
4. **Release hygiene:** example datasets (within patient-data rules) + pre-trained checkpoints + a documented `--model-type` recipe per biomarker, so the "forked, openly available" promise is real.

**Done when:** one command produces a held-out per-class report, and one produces the
time-to-accuracy curve for a run.

---

## 8. Suggested sequencing (maps to the proposal's 7-month timeline)

The proposal's timeline is: Foundation backbone (1mo, **done**) → curriculum + backends (3mo) →
LoRA (1mo) → evaluation & user testing (2mo). Slot the work like this:

1. **Now (days):** §3 prerequisite fixes — NaN-F1 (3.1), prev-model caching (3.2), quick correctness (3.5). These are cheap and unblock honest measurement.
2. **Backbone-polish + eval bootstrap (weeks):** §7 test scorer (7.1) + the sampler refactor (3.3) and patient-clean split (3.4). You need the scorer before you can claim any curriculum/LoRA improves anything.
3. **Curriculum (the proposal's 3-month block):** §6, built on the §3.3 sampler. Start with easy/hard real-data staging; treat synthetic-lesion generation (6.4) as a parallel research track.
4. **LoRA (the proposal's 1-month block):** §5 — pairs naturally with the prev-model-caching win.
5. **Multi-class (largest, schedule deliberately):** §4 — crosses into the painter, so plan the annotation-format change carefully; it can proceed in parallel once the encoding (4.1) is fixed.
6. **Evaluation & user testing (the proposal's final 2-month block):** §7.2–7.4 + usability.

---

## 9. Open questions for Donna (would sharpen this plan)
1. **Multi-class scope:** is one combined RIPL+SDD(+Drusen) model a near-term deliverable, or is the single-biomarker RETFound model the current priority? P2 is by far the biggest lift because it touches the painter.
2. **Painter changes allowed?** P2 (class brushes) and P4 (easy/hard tagging) both need small painter additions. CLAUDE.md currently treats the painter as frozen — confirm we can modify it, or we route these through sidecar files instead.
3. **Synthetic lesions:** do you already have a method/source for synthetic RIPL/SDD insertion (Stage 1), or should that be scoped as its own research subtask?
4. **Day-to-day model type:** `retfound` vs `retfound_rfa` vs `unet` right now — this sets which prerequisite fixes (esp. 3.2) bite hardest.
