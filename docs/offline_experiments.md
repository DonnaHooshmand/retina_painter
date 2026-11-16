# Offline training experiments

Offline experiments train a pre-populated project to convergence without the
painter and score the result against a curated dataset's reference labels.
They answer model-comparison questions (does `retfound_rfa` beat `unet`? do
dense reviewed masks help?). They do not test the interactive loop — that
still needs the painter.

The tools live in `trainer/experiments/` and run from any directory with the
trainer virtual environment active.

| Script | Purpose |
|---|---|
| `prepare_offline_project.py` | Build a painter-compatible project from a curated dataset (sparse strokes or dense reviewed masks) |
| `run_offline_training.py` | Train the project with the production `Trainer` loop until early stopping |
| `evaluate_project.py` | Run the painter's inference path and report scan-, lesion- and pixel-level metrics |
| `compare_evaluations.py` | Paired comparison of two evaluations with patient-level bootstrap intervals |
| `run_cross_validation.py` | Prepare, train and score a patient-level cross-validation run in one command |
| `score_cross_validation.py` | Blob-level scores for a cross-validation run, pooled over folds, with patient-level intervals |
| `run_session_replay.py` | Replay an annotation session on one fold: annotations arrive one scan at a time while the production trainer runs |
| `score_session_replay.py` | Score the model on screen at each milestone of a replayed session on the held-out patients |
| `prepare_error_review.py` | Collect the scans several models get wrong into a small project to look at in the painter |
| `mask_io.py` | The one correct way to read reference masks and read/write annotations |

## Workflow

```powershell
cd trainer\experiments
$ds = "D:\RootPainterSync\curated_datasets\Combined_Development_20261105_v1"

# 1. One project per arm. Same seed, same data; only the model differs.
python prepare_offline_project.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name combined_sparse_retfound_rfa_seed0 --model-type retfound_rfa --seed 0
python prepare_offline_project.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name combined_sparse_unet_seed0 --model-type unet --seed 0

# 2. Train. One GPU per run, so two arms train at the same time.
python -u run_offline_training.py --project D:\RootPainterSync\projects\combined_sparse_retfound_rfa_seed0 --gpu 0
python -u run_offline_training.py --project D:\RootPainterSync\projects\combined_sparse_unet_seed0 --gpu 1

# 3. Score the validation-best checkpoint.
python evaluate_project.py --project D:\RootPainterSync\projects\combined_sparse_retfound_rfa_seed0 --gpu 0
python evaluate_project.py --project D:\RootPainterSync\projects\combined_sparse_unet_seed0 --gpu 1

# 4. Paired comparison (B minus A).
python compare_evaluations.py `
    --a D:\RootPainterSync\projects\combined_sparse_unet_seed0\evaluation\val__<checkpoint> `
    --b D:\RootPainterSync\projects\combined_sparse_retfound_rfa_seed0\evaluation\val__<checkpoint>
```

Projects are created under `<sync>/projects/` with a normal `.seg_proj`, so
they open in the painter. Add `--write-project-segmentations --split all` to
`evaluate_project.py` to pre-fill `segmentations/`; the painter then shows the
final checkpoint's overlay on every scan without a running trainer.

## What the training run is

`run_offline_training.py` subclasses the production `Trainer` and calls
`start_training` / `train_one_epoch` directly. Sampler, loss, optimizer,
validation, checkpoint promotion and early stopping are the painter
workflow's own code. Four things differ; none changes a training
decision:

- The shared `<sync>/instructions` queue is neither cleared at start-up nor
  polled, so an open painter cannot switch the model mid-run and several runs
  can train in parallel.
- Event logs go to `<project>/logs/offline_server_log.txt` instead of the
  shared `server_log.txt`; painter status messages are not written.
- `--gpu` pins the run to one device.
- The best checkpoint is scored on the validation set once, and the result is
  reused until the checkpoint or the validation annotations change. The
  interactive trainer reloads and re-scores it every epoch because a clinician
  may have edited the annotations in between. With fixed annotations the result
  is bit-identical every time (checked on 271 logged epochs across three runs),
  so the cache only saves time: about 27 s of each 92 s RETFound epoch.
  `--no-validation-cache` restores per-epoch re-scoring.

### Rollback is off by default

Training is conventional (the original RootPainter protocol): the candidate
trains continuously, the best validation checkpoint is the one kept, and early
stopping decides when to stop. This is also the live trainer's default since
2026-11-13.

Automatic candidate rollback used to be on: after three consecutive
validation epochs worse than the best checkpoint, training restarted from that
checkpoint with a fresh optimizer. On annotations that are not changing that
turns training into repeated three-epoch attempts from the best checkpoint. It
fired at least 26 times in one three-hour RETFound-RFA run, and a U-Net never
left its first plateau with it on but trained with it off. While annotations
are arriving it rarely fires at all, because its count restarts with every
change, so it does its damage exactly when annotation pauses.

`--candidate-rollback-patience N` turns it on for a controlled comparison (on
`run_offline_training.py`, `run_session_replay.py`, `main.py` and
`start-trainer`). `--no-rollback` is still accepted and changes nothing. The
run summary records the setting and the number of rollbacks.

### Time budgets and the original RootPainter baseline

`--max-minutes N` ends a run after the first epoch that finishes beyond N
minutes of training (model loading is not counted). Use it to compare what
each configuration delivers in the same wall-clock time, which is the question
an interactive tool has to answer.

`--trainer-commit <sha>` runs the trainer source exactly as it was at that
commit. The source is exported with `git archive` into
`<project>/trainer_src/<sha>/` and driven unmodified: no validation cache, and
options the older trainer lacks (model type, loss type, rollback) are left
out. `4ffbda3` is the last upstream RootPainter commit before the RETFound
work, so this is the RootPainter baseline:

```powershell
python prepare_offline_project.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name combined_sparse_rootpainter_original_seed0 --model-type unet --seed 0
python -u run_offline_training.py `
    --project D:\RootPainterSync\projects\combined_sparse_rootpainter_original_seed0 `
    --gpu 0 --trainer-commit 4ffbda3 --max-minutes 45
```

What that commit does differently from the current trainer:

| | RootPainter `4ffbda3` | Current trainer |
|---|---|---|
| Unpainted pixels | Logits zeroed, so they sit at probability 0.5 and are averaged into the loss | Excluded from the loss |
| Checkpoint kept | Highest validation F1 | Lowest masked Dice + 0.3 CE validation loss |
| Rollback | None | None by default (optional, `--candidate-rollback-patience N`) |
| Training crops | Random crop of a random annotated scan | Crop must contain a stroke of the sampled class |
| Seeding | None | Trial seed |

The loss formula (Dice + 0.3 cross-entropy) and the U-Net optimiser (SGD,
learning rate 0.01, momentum 0.99) are the same. Its checkpoints load in
`evaluate_project.py` as `unet`.

The original trainer has no seed handling, so the runner supplies it: it
seeds the main process (model initialisation) and gives the trainer's
DataLoader a seeded generator and per-worker seeding. That changes which
random numbers are drawn and nothing about the algorithm. Two seeded
two-epoch runs on the two GPUs produced identical validation metrics and
training losses equal to ten significant figures.

An "epoch" is the trainer's usual 612 random crops followed by a full pass
over `annotations/val`. Training ends after `--max-epochs-without-progress`
(default 60) validation epochs with no improvement in the masked Dice + 0.3 CE
validation loss. The latest file in `models/` is the validation-best
checkpoint. Create an empty file named `STOP` in the project to end a run
early; re-run the command to resume from the latest checkpoint.

## Reading the evaluation

Three views, from most to least trustworthy reference:

1. **B-scan level.** A scan is called positive when its largest predicted
   connected component has at least `--min-lesion-px` pixels (default 25, fixed
   in advance rather than tuned). The reference is the clinician's
   present/absent decision. This is the clinical endpoint of the earlier
   RootPainter RIPL study. Sensitivity and specificity come with Wilson 95%
   intervals; AUROC is threshold-free.
2. **Lesion level.** Fraction of reference lesions touched by a prediction, and
   false-positive blobs per scan. Components under `--min-lesion-px` are
   ignored on both sides.
3. **Pixel level.** Pooled precision / recall / F1 against
   `ground_truth/reviewed_masks`.

Uncertain scans are excluded from every metric.

Caveats that apply to every number:

- **Validation is not a test set.** The checkpoint is selected on the
  validation strokes, so validation metrics are optimistic. Both arms share the
  bias, which makes the comparison fair but not the absolute level. The
  combined dataset has no held-out test cohort (`final_test_created: false`).
- **Small positive count.** The validation split has 19 positive scans, so one
  scan moves sensitivity by about five points. Use `compare_evaluations.py`
  and read the interval, not the point estimate.
- **Reviewed masks are corrected predictions.** They are exact on absent scans
  but can keep uncorrected output from an earlier model on present scans.

## Cross-validation: the yardstick for comparing changes

A single train/validation split of this dataset has 19 positive validation
scans. One scan is five points of sensitivity, and the checkpoint was chosen
on those same scans. That cannot rank ideas: four RETFound-RFA configurations
and two long runs all found between 8 and 11 of the 19 at matched specificity.
Cross-validation scores every scan with a model that never saw its patient,
so all 103 positive scans count.

```powershell
python run_cross_validation.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name cv_sparse_retfound_rfa --model-type retfound_rfa --folds 4 `
    --gpus 0,1 --max-minutes 45
```

Everything after a `--` goes to `run_offline_training.py`, so a configuration is
its preparation options (`--supervision`, `--model-type`) plus its training
options. The command builds `<name>_fold0` … (via
`prepare_offline_project.py --folds K --fold k`), trains one fold per GPU at a
time, then scores. Re-running it resumes. With four folds on two GPUs a run
takes two training budgets plus model loading and scoring (about 1 h 55 min at
45 minutes per fold for RETFound). Add `--baseline <other run>` for paired
differences.

**Folds.** `patient_folds` deals patients into K groups, patients with a scan
decided present separately from the rest, so positives are spread evenly. For
fold k those patients are held out and get no annotation at all. The remaining
patients are split again by patient, one part in six becoming the validation
set the trainer uses for checkpoint choice. No patient has two roles, and over
the K folds every scan is held out exactly once. The dataset's own `split`
column is not used.

**Scores are about blobs, not pixels.** A RIPL is either there or not, and the
clinician marks it with a dot (most marks are one 25-pixel brush dab), so
pixel overlap with the marks says little. Per scan the model gets a *blob
score*: the highest probability threshold at which its prediction still holds
a connected blob of at least 25 pixels.

| Reported | Meaning |
|---|---|
| AUROC of the blob score | How well scans decided present rank above scans decided absent, no threshold chosen |
| Positive scans flagged, negative scans cleared | At each fold's own threshold, set on that fold's *validation* scans to clear 90% of the absent ones, then applied unchanged to the held-out scans |
| Clinician marks found | Red marks touched by a predicted blob at that threshold |
| False-positive blobs per negative scan | Predicted blobs on scans decided absent |

Intervals are patient-level bootstrap intervals. Per-scan rows are written to
`<fold project>/evaluation/fold__<checkpoint>/fold_scores.csv` and never
printed; the pooled summary goes to
`<sync>/projects/<name>_cross_validation.json`.

Because the threshold is set on validation scans and applied to held-out
ones, the held-out specificity is an honest estimate, not 90% by construction.

The first baseline (`cv_sparse_retfound_rfa`: RETFound-RFA, sparse strokes,
no rollback, 45 minutes per fold) flags 77 of 103 positive scans and
clears 346 of 387 negative ones, AUROC 0.87 [0.82, 0.91]. At the painter's
fixed 0.5 cut-off the same models flag 84 of 103 and clear 307 of 387.

## Session replay: the test that matches the task

Cross-validation above trains on every annotation at once. That is the end of
a session. RetinaPainter's task is the session itself: the model is trained
while the user works through scans it has not seen. A replayed session
measures that with the annotations that already exist, so nobody has to sit at
the painter for each experiment.

```powershell
python run_cross_validation.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name replay_retfound_rfa --model-type retfound_rfa --folds 4 `
    --gpus 0,1 --session-pace 20 -- --milestones 25,50,100,200
```

`--session-pace` (seconds the user spends on each scan) switches the driver to
replay mode. Options after `--` go to `run_session_replay.py`: the trainer
options of `run_offline_training.py` (for example `--loss-type`) plus
`--milestones`, `--warmup-scans`, `--max-scans`, `--tail-minutes`,
`--validation-cache` and `--no-segmentation`. `--max-minutes` does not apply;
a session lasts `scans x pace`. With `--baseline <other replay>` the scorer
adds paired differences at every milestone both runs recorded.

**What is replayed.** Each fold project is prepared with `--session-split`:
the fold's patients are held out exactly as in cross-validation, and the
remaining scans are put in the painter's seeded navigation order and split by
the painter's own patient-aware rule (`patient_grouped_train_val_split`,
loaded from the painter source), so validation scans arrive throughout the
session as they do for a user. Then, per fold:

1. Every annotation is hidden (`<project>/replay/pending/`).
2. Scan *i* is finished `i x pace` seconds into the session. At that moment
   its annotation file, if the clinician painted on it, appears on its side.
   Unpainted scans take the same time and add nothing, as in the painter.
3. Training starts after the first 10 scans (`auto_start_training_after_reviews`),
   as the painter starts it. The production `Trainer` then runs unchanged:
   new annotations are picked up at the start of each epoch, and validation,
   checkpoint promotion and early stopping behave as they do live.
   The best checkpoint is re-scored every epoch, as live (the offline cache is
   off unless `--validation-cache`).
4. For each scan that comes on screen the trainer draws the prediction through
   its own `segment` path, because the painter asks for one per scan and that
   time is not spent training.
5. At each milestone (scans viewed) and at the end, the model the painter would
   be showing is recorded: the live weights while validation has no lesion
   mark yet, the best checkpoint afterwards. This uses the trainer's own
   choice (`_get_provisional_live_model_for_segment`).

Session time is real time. Stretches where the live trainer would be idle
(before training starts, or while one side has no annotation) are skipped, not
slept through, and model loading is not counted.

**Scores.** `score_session_replay.py` scores each recorded model on that
fold's held-out patients and pools the folds milestone by milestone. A session
has almost no validation scans early on, so no cut-off can be calibrated the
way cross-validation does it. Reported per milestone: scans viewed, session
minutes, annotated scans (and how many carry a lesion mark), AUROC of the blob
score with a patient-level interval, positive scans flagged / negative scans
cleared / clinician marks found / false blobs per negative scan at the
painter's 0.5 cut-off (what the user sees), and positive scans flagged at the
cut-off that clears 90% of the held-out negatives. That last column picks its
cut-off on the scans it is reported on; it describes ranking at a fixed
strictness and is blank when no cut-off clears that many. The pooled table goes
to `<sync>/projects/<name>_session_replay.json`, per-scan rows to
`<fold project>/evaluation/replay__<model>/held_out_scores.csv`.

**Limits.**

- The stored strokes corrected an earlier model's mistakes, not this one's. A
  replay therefore approximates a corrective session; a person timed at the
  painter remains the real check.
- The pace is a setting (default 20 s per scan). The dataset records no
  per-scan timing, so set it from a timed session when one exists.
- Milestones are recorded at the first epoch boundary after the scan count is
  reached, so the model can be up to one epoch newer than at that instant.
- A replay needs a fresh project and cannot be resumed half way; delete the
  fold's project folder and re-run. Scans beyond `--max-scans` stay in
  `replay/pending/`.

**First baseline** (`replay_sparse_retfound_rfa`, 2026-11-13: RETFound-RFA,
sparse strokes, no rollback, 20 s per scan, first 200 scans of each fold,
490 held-out scans from 160 patients, 103 positive and 387 negative). Model on
screen, 0.5 cut-off:

| Scans viewed | Minutes | Annotated (with lesion mark) | AUROC | Positive flagged | Negative cleared | False blobs per negative scan |
|---|---|---|---|---|---|---|
| 25 | 9 | 20 (7) | 0.49 [0.44, 0.55] | 102/103 | 1/387 | 17.9 |
| 50 | 17 | 38 (13) | 0.55 [0.50, 0.60] | 102/103 | 0/387 | 16.5 |
| 100 | 34 | 71 (24) | 0.59 [0.54, 0.64] | 102/103 | 4/387 | 11.1 |
| 150 | 50 | 99 (31) | 0.69 [0.63, 0.75] | 102/103 | 4/387 | 7.3 |
| 200 | 67 | 128 (37) | 0.70 [0.64, 0.76] | 92/103 | 102/387 | 3.6 |

For 50 minutes the model puts a blob on nearly every scan, and after 67 minutes
it is still far from the all-at-once figure above (AUROC 0.87, 307 of 387
cleared at 0.5). The sessions ran more epochs than the cross-validation folds
(57 to 67 against 36 to 44), so this is not a shortage of training steps. They
had half the annotations (128 against 249 per fold; 72 of the 200 scans viewed
carried no stroke and taught the model nothing), and the best checkpoint does
well on the validation strokes (recall 0.72 to 0.93) while painting on the
unpainted retina of held-out scans: only painted pixels are supervised, and 6
to 9% of those are lesion. No cut-off up to 0.99 clears 90% of the negatives in
every fold. Results differ by fold at the end (negatives cleared: 1/97, 60/106,
33/92, 8/92).

**Absent decisions as whole clean scans** (`replay_sparse_absent_retfound_rfa`,
same day, same folds, scan order and split; `--supervision sparse_absent`).
About 190 of the 200 scans viewed then carry supervision instead of 128. The
painter writes the same files for a project created with **Use scans marked
RIPL absent as clean scans** (off by default; see
`docs/clinician_review_workflow.md`).

| Scans viewed | AUROC | Paired AUROC difference | Positive flagged | Negative cleared | False blobs per negative scan |
|---|---|---|---|---|---|
| 25 | 0.53 | +0.04 [-0.03, +0.10] | 90/103 | 58/387 | 3.1 |
| 50 | 0.64 | +0.08 [+0.01, +0.15] | 87/103 | 88/387 | 3.7 |
| 100 | 0.68 | +0.08 [+0.02, +0.15] | 81/103 | 163/387 | 1.3 |
| 150 | 0.73 | +0.04 [-0.03, +0.10] | 70/103 | 273/387 | 0.5 |
| 200 | 0.78 | +0.08 [+0.01, +0.14] | 69/103 | 293/387 | 0.3 |

The blob-on-every-scan failure is gone from the first milestone and ranking is
a little better throughout, but lesion scans are lost: 69 of 103 flagged at
200 scans against 92, and some lesion scans now hold no blob at any cut-off
(7 at 100 scans, 22 at 150, 14 at 200; none in the baseline), so a lower
display cut-off cannot bring them back. The supervised pixels swung from 6 to
9% lesion to about 0.2%, and the adaptive sampler's lesion-crop share fell to
17 to 18% of each epoch (24 to 26% in the baseline) as clean scans accumulated.
Still short of the all-at-once figure, and in two folds the model on screen
did not change between two milestones (about 15 epochs without a promotion).

**Absent decisions as clean scans, lesion crops held at half of each epoch**
(`replay_sparse_absent_fg50_retfound_rfa`, 2026-11-14, commit `ea7bf36`; same
folds, scan order and split; `--supervision sparse_absent` with
`--foreground-tile-fraction 0.5` after `--`). Every epoch of every fold logged
`foreground tile target 50% (fixed)`.

| Scans viewed | AUROC | Paired AUROC difference vs sparse | vs absent, adaptive | Positive flagged | Negative cleared | False blobs per negative scan | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|---|
| 25 | 0.49 [0.43, 0.55] | -0.01 [-0.08, +0.06] | -0.04 [-0.09, +0.01] | 100/103 | 5/387 | 3.9 | 0 |
| 50 | 0.55 [0.49, 0.61] | 0.00 [-0.06, +0.07] | -0.08 [-0.16, 0.00] | 80/103 | 108/387 | 2.5 | 5 |
| 100 | 0.74 [0.69, 0.79] | +0.15 [+0.10, +0.21] | +0.07 [+0.02, +0.12] | 93/103 | 142/387 | 1.4 | 2 |
| 150 | 0.81 [0.76, 0.86] | +0.12 [+0.06, +0.18] | +0.08 [+0.03, +0.14] | 83/103 | 266/387 | 0.45 | 5 |
| 200 | 0.84 [0.80, 0.89] | +0.14 [+0.09, +0.19] | +0.07 [+0.02, +0.12] | 91/103 | 277/387 | 0.38 | 7 |

At 200 scans the fixed share brings back most of the lesion scans the clean
scans cost (91 of 103 flagged against 69, +21 points [+11, +32]) and keeps
most of the clearing (277 of 387 against 293, -4 points [-11, +2]). Against the
sparse baseline it flags as many lesion scans (91 against 92) and clears 277
clean scans instead of 102. The end-of-session model is close to the
all-at-once figure (AUROC 0.87; 84 of 103 and 307 of 387 at 0.5), from 190
annotated scans in 67 minutes.

What it does not fix:

- **The first 50 scans (17 minutes).** Ranking is no better than chance or the
  sparse baseline (0.49, 0.55), and at 25 scans the model flags nearly every
  scan again (5 of 387 cleared; the adaptive share cleared 58). The
  early-session test below puts this down to training time, not annotations.
- **The picture at the 0.5 cut-off is not steady.** Lesion scans flagged go
  100, 80, 93, 83, 91 across the milestones while ranking rises every time,
  and single folds swing between milestones (one clears 70 of 97 clean scans
  at 150 scans and 49 at 200; another 44 of 106 at 100 and 91 at 200). The
  model on screen is chosen on 4 to 8 validation scans with a lesion mark.
- **Seven lesion scans still hold no blob at any cut-off** at 200 scans (14
  with the adaptive share, none in the sparse baseline).
- **Pixel balance is still lopsided.** Lesion pixels are 0.8 to 0.9% of the
  supervised pixels in the training crops (0.2 to 0.3% with the adaptive
  share, 8 to 11% with sparse strokes), because a clean crop is supervised
  everywhere and a lesion crop only on its strokes.

Promotions were more frequent (21 to 24 per fold against 15 to 20; longest
stretch without one 9 to 11 epochs against 11 to 21), and no milestone showed
the previous milestone's checkpoint. The intervals cover which patients were
held out, not run-to-run variation in training; the second seed below measures
that.

**The same setting on a second trial seed**
(`replay_sparse_absent_fg50_seed1_retfound_rfa`, 2026-11-14, `--seed 1`: other
patient folds, scan order, split, initial decoder weights and sampling; working
tree at `ea7bf36` plus the lesion-crop-share change, the share arriving in the
instruction). Both runs score the same 490 held-out scans, each scan by a model
that never saw its patient, so the difference is paired.

| Scans viewed | AUROC, seed 0 | AUROC, seed 1 | Paired difference (seed 1 minus seed 0) | Positive flagged, seed 0 / seed 1 | Negative cleared, seed 0 / seed 1 | False blobs per negative scan | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|---|
| 25 | 0.49 | 0.49 [0.44, 0.55] | 0.00 [-0.08, +0.08] | 100 / 97 | 5 / 21 | 3.9 / 4.6 | 0 / 0 |
| 50 | 0.55 | 0.65 [0.58, 0.72] | +0.09 [-0.01, +0.19] | 80 / 83 | 108 / 133 | 2.5 / 2.2 | 5 / 4 |
| 100 | 0.74 | 0.72 [0.66, 0.77] | -0.03 [-0.09, +0.04] | 93 / 92 | 142 / 172 | 1.4 / 1.6 | 2 / 6 |
| 150 | 0.81 | 0.78 [0.71, 0.83] | -0.04 [-0.09, +0.02] | 83 / 83 | 266 / 230 | 0.45 / 0.64 | 5 / 9 |
| 200 | 0.84 | 0.78 [0.72, 0.83] | -0.06 [-0.11, -0.02] | 91 / 87 | 277 / 241 | 0.38 / 0.56 | 7 / 9 |

What repeats: no better than chance at 25 scans with nearly every scan
flagged; AUROC 0.72 to 0.74 at 100 scans; and at 200 scans 84 to 88% of lesion
scans flagged with 62 to 72% of clean scans cleared. The unsteady picture at
the 0.5 cut-off repeats too (lesion scans flagged 97, 83, 92, 83, 87; one fold
clears 65 of 87 clean scans at 50 scans, 7 at 100 and 62 at 200), and one fold
showed the same checkpoint at 150 and 200 scans after 20 epochs without a
promotion.

What does not repeat: the end of the session. The second seed ranks 0.06 lower
[-0.11, -0.02] and clears 9 points fewer clean scans [-16, -3] (flagged -4
points [-13, +4]). **Two runs of one setting can differ by about 0.06 AUROC and
9 points of clearing, so a single-run difference of that size between settings
is not evidence.** Each step of the setting is therefore compared within one
seed, on the same sessions, below.

**The adaptive share on the second seed**
(`replay_sparse_absent_seed1_retfound_rfa`, 2026-11-14: the seed-1 sessions
again, same annotations, without the fixed share. Every sampler line reads
`(adaptive)`; the target ranged from 9 to 31% and ended at 15 to 19%.)

| Scans viewed | AUROC, adaptive | AUROC, fixed half | Paired difference (half minus adaptive) | Positive flagged, adaptive / half | Negative cleared, adaptive / half | False blobs per negative scan | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|---|
| 25 | 0.55 [0.50, 0.61] | 0.49 | -0.06 [-0.13, +0.01] | 95 / 97 | 45 / 21 | 3.4 / 4.6 | 0 / 0 |
| 50 | 0.55 [0.49, 0.62] | 0.65 | +0.09 [-0.01, +0.20] | 83 / 83 | 76 / 133 | 2.7 / 2.2 | 0 / 4 |
| 100 | 0.59 [0.52, 0.65] | 0.72 | +0.13 [+0.06, +0.20] | 52 / 92 | 241 / 172 | 0.91 / 1.6 | 39 / 6 |
| 150 | 0.79 [0.73, 0.85] | 0.78 | -0.02 [-0.08, +0.04] | 68 / 83 | 341 / 230 | 0.14 / 0.64 | 25 / 9 |
| 200 | 0.77 [0.71, 0.84] | 0.78 | 0.00 [-0.05, +0.05] | 62 / 87 | 342 / 241 | 0.15 / 0.56 | 30 / 9 |

The fixed half against the adaptive share at 200 scans, same sessions:

| | Seed 0 | Seed 1 |
|---|---|---|
| AUROC | +0.07 [+0.02, +0.12] | 0.00 [-0.05, +0.05] |
| Lesion scans flagged | 91 against 69, +21 points [+11, +32] | 87 against 62, +24 points [+16, +33] |
| Clean scans cleared | 277 against 293, -4 points [-11, +2] | 241 against 342, -26 points [-31, -21] |
| Lesion scans with no blob at any cut-off | 7 against 14 | 9 against 30 |

On both seeds the fixed half flags 21 to 24 points more lesion scans and
leaves far fewer lesion scans with no blob at any cut-off; with the adaptive
share on seed 1, 30 of the 103 lesion scans cannot be shown at any display
cut-off, and at 100 scans it flags only 52. The cost is clean scans cleared, 4
to 26 points. Ranking at the end is better on one seed and equal on the other;
at 100 scans it is better on both (+0.07 and +0.13). The fixed half is
therefore a move toward sensitivity that costs no ranking, not a ranking gain
that can be counted on.

**Strokes only on the second seed, and the three settings on both seeds**
(`replay_sparse_seed1_retfound_rfa`, 2026-11-14: the seed-1 sessions with
sparse strokes only, 117 to 126 annotated scans per fold instead of 185 to
191.) Strokes only at 25, 50, 100, 150 and 200 scans: AUROC 0.48, 0.51, 0.58,
0.68, 0.69; lesion scans flagged 76, 103, 92, 93, 92 of 103; clean scans
cleared 87, 0, 50, 120, 135 of 387. It repeats the first baseline (seed 1
minus seed 0 at 200 scans: AUROC -0.02 [-0.08, +0.05], flagged 0 points,
cleared +9 points [-1, +18]).

Model on screen after 200 scans (67 minutes), 0.5 cut-off, 103 lesion scans
and 387 clean scans held out, seed 0 / seed 1:

| Setting | AUROC | Lesion scans flagged | Clean scans cleared | False blobs per clean scan | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|
| Strokes only | 0.70 / 0.69 | 92 / 92 | 102 / 135 | 3.6 / 4.5 | 0 / 0 |
| + absent decisions as clean scans | 0.78 / 0.77 | 69 / 62 | 293 / 342 | 0.3 / 0.15 | 14 / 30 |
| + lesion crops held at half | 0.84 / 0.78 | 91 / 87 | 277 / 241 | 0.38 / 0.56 | 7 / 9 |

Paired differences on the same sessions at 200 scans:

| Comparison | Seed | AUROC | Lesion scans flagged | Clean scans cleared |
|---|---|---|---|---|
| Full setting minus strokes only | 0 | +0.14 [+0.09, +0.19] | -1 point [-8, +6] | +45 points [+39, +51] |
| | 1 | +0.09 [+0.03, +0.16] | -5 points [-13, +2] | +27 points [+19, +36] |
| Clean scans alone minus strokes only | 0 | +0.08 [+0.01, +0.14] | -22 points | +49 points |
| | 1 | +0.09 [+0.03, +0.16] | -29 points [-40, -19] | +53 points [+46, +61] |
| Fixed half minus adaptive share | 0 | +0.07 [+0.02, +0.12] | +21 points [+11, +32] | -4 points [-11, +2] |
| | 1 | 0.00 [-0.05, +0.05] | +24 points [+16, +33] | -26 points [-31, -21] |

What holds on both seeds:

- The full setting ranks better than strokes alone (+0.09 to +0.14), clears 27
  to 45 points more clean scans, and puts 0.4 to 0.6 false blobs on a clean
  scan instead of 3.6 to 4.5, without flagging significantly fewer lesion
  scans.
- Clean scans alone buy the clearing but lose 22 to 29 points of lesion scans,
  and 14 to 30 lesion scans then hold no blob at any cut-off. The fixed half is
  what gets them back.
- Strokes only never loses a lesion scan entirely, and clears few clean scans
  at any point of the session.

What is still open after these runs: the full setting's end of session varies
between runs (AUROC 0.78 and 0.84, clean scans cleared 241 and 277), the
picture at the 0.5 cut-off is unsteady between milestones in every setting,
and two seeds are not a distribution. A reported table needs at least a third
seed per row; that seed follows.

**The three settings on a third seed, and the three-seed table**
(`replay_sparse_absent_fg50_seed2_retfound_rfa`,
`replay_sparse_absent_seed2_retfound_rfa`, `replay_sparse_seed2_retfound_rfa`,
2026-11-15, `--seed 2`, commit `08ab2bf` with a clean working tree. In all
twelve folds no patient is on two sides, no held-out scan was annotated, every
sampler line reads `(fixed)` at 50% or `(adaptive)` as intended, and rollback
is off.)

Seed 2 through the session (AUROC; lesion scans flagged of 103; clean scans
cleared of 387):

| Scans viewed | Strokes only | + absent decisions as clean scans | + lesion crops held at half |
|---|---|---|---|
| 25 | 0.47; 48; 197 | 0.53; 79; 119 | 0.52; 99; 34 |
| 50 | 0.59; 97; 28 | 0.57; 91; 67 | 0.58; 87; 99 |
| 100 | 0.63; 100; 21 | 0.64; 78; 173 | 0.68; 80; 188 |
| 150 | 0.78; 93; 118 | 0.72; 55; 328 | 0.73; 92; 150 |
| 200 | 0.75 [0.69, 0.81]; 82; 182 | 0.75 [0.69, 0.80]; 67; 300 | 0.78 [0.73, 0.83]; 80; 242 |

Model on screen after 200 scans (67 minutes), 0.5 cut-off, seed 0 / seed 1 /
seed 2, then the average over the three seeds. The interval resamples
patients, with the same resample applied to every run; it covers which
patients were held out, not what a fourth seed would add, so read it beside
the three per-seed values.

| Setting | AUROC | Lesion scans flagged (of 103) | Clean scans cleared (of 387) | False blobs per clean scan | Clinician marks found | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|
| Strokes only | 0.70 / 0.69 / 0.75; mean 0.71 [0.67, 0.76] | 92 / 92 / 82; 86% [81, 91] | 102 / 135 / 182; 36% [32, 40] | 3.6 / 4.5 / 1.5; mean 3.2 | 68% | 0 / 0 / 0 |
| + absent decisions as clean scans | 0.78 / 0.77 / 0.75; mean 0.77 [0.72, 0.81] | 69 / 62 / 67; 64% [55, 73] | 293 / 342 / 300; 81% [77, 84] | 0.33 / 0.15 / 0.28; mean 0.25 | 60% | 14 / 30 / 25; mean 23 |
| + lesion crops held at half | 0.84 / 0.78 / 0.78; mean 0.80 [0.76, 0.84] | 91 / 87 / 80; 83% [77, 89] | 277 / 241 / 242; 65% [62, 70] | 0.38 / 0.56 / 0.50; mean 0.48 | 78% | 7 / 9 / 8; mean 8 |

Paired differences on the same sessions at 200 scans, seed 0 / seed 1 /
seed 2, then the average over seeds with its patient-level interval:

| Comparison | AUROC | Lesion scans flagged, points | Clean scans cleared, points | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|
| Full setting minus strokes only | +0.14 / +0.09 / +0.04; mean +0.09 [+0.06, +0.12] | -1 / -5 / -2; mean -3 [-8, +2] | +45 / +27 / +16; mean +29 [+25, +34] | +7 / +9 / +8 |
| Clean scans alone minus strokes only | +0.08 / +0.09 / 0.00; mean +0.05 [+0.01, +0.09] | -22 / -29 / -15; mean -22 [-30, -15] | +49 / +53 / +30; mean +44 [+40, +49] | +14 / +30 / +25 |
| Fixed half minus adaptive share | +0.07 / 0.00 / +0.04; mean +0.04 [0.00, +0.07] | +21 / +24 / +13; mean +19 [+13, +26] | -4 / -26 / -15; mean -15 [-18, -12] | -7 / -21 / -17 |

On seed 2 alone: full setting minus strokes only, AUROC +0.04 [-0.01, +0.08],
flagged -2 points [-10, +5], cleared +16 points [+9, +22]; clean scans alone
minus strokes only, 0.00 [-0.06, +0.05], -15 points [-24, -5], +30 points
[+24, +37]; fixed half minus adaptive, +0.04 [-0.02, +0.10], +13 points
[+3, +22], -15 points [-21, -9].

What holds on all three seeds:

- **The full setting clears more clean scans than strokes alone and flags
  about as many lesion scans.** 16 to 45 points more cleared (mean +29),
  0.4 to 0.6 false blobs on a clean scan instead of 1.5 to 4.5, and a
  difference in lesion scans flagged that is not significant on any seed
  (mean -3 points [-8, +2]). It also finds more of the clinician's marks
  (78% against 68%, +10 points [+5, +15]).
- **Clean scans alone lose lesion scans, and the fixed half gets them back.**
  Without the fixed half, 15 to 29 points fewer lesion scans are flagged than
  with strokes alone and 14 to 30 hold no blob at any cut-off; with it, 13 to
  24 points come back and 7 to 9 stay blank, for 4 to 26 points of clearing.
- **The fixed half costs no ranking** (+0.07, 0.00, +0.04).

What the third seed changes:

- **The ranking gain over strokes alone is smaller than two seeds suggested.**
  Strokes only did better on seed 2 (AUROC 0.75, 182 clean scans cleared) than
  on seeds 0 and 1, and on that seed the full setting's ranking gain is +0.04
  with an interval that includes zero. Over three seeds it is +0.09
  [+0.06, +0.12]; quote that, not the +0.14 of seed 0.
- **Seed 0 was the high run of the full setting.** Seeds 1 and 2 agree (AUROC
  0.78 both, 241 and 242 clean scans cleared); the three-seed figure is AUROC
  0.80, 83% of lesion scans flagged and 65% of clean scans cleared.
- **Lesion scans flagged at 0.5 also varies between runs**: 91, 87 and 80 for
  the full setting (seed 2 minus seed 0, -11 points [-19, -3]), and 92, 92
  and 82 for strokes only.
- **Strokes only is the least repeatable row for clearing**: 102, 135 and 182
  of 387.

Earlier in the session, averaged over the three seeds (AUROC; lesion scans
flagged; clean scans cleared):

| Scans viewed | Strokes only | + absent decisions as clean scans | + lesion crops held at half | Full setting minus strokes only, AUROC |
|---|---|---|---|---|
| 25 | 0.48; 73%; 25% | 0.54; 85%; 19% | 0.50; 96%; 5% | +0.02 [-0.03, +0.06] |
| 50 | 0.55; 98%; 2% | 0.59; 84%; 20% | 0.59; 81%; 29% | +0.04 [0.00, +0.08] |
| 100 | 0.60; 95%; 6% | 0.63; 68%; 50% | 0.71; 86%; 43% | +0.11 [+0.08, +0.15] |
| 150 | 0.72; 93%; 21% | 0.75; 62%; 81% | 0.77; 83%; 56% | +0.06 [+0.02, +0.09] |
| 200 | 0.71; 86%; 36% | 0.77; 64%; 81% | 0.80; 83%; 65% | +0.09 [+0.06, +0.12] |

No setting ranks better than chance at 25 scans, and all three sit at 0.55 to
0.59 at 50; the settings separate from 100 scans (33 minutes) on. The picture
at the 0.5 cut-off is still unsteady within a session on seed 2 (full setting:
lesion scans flagged 99, 87, 80, 92, 80; clean scans cleared 34, 99, 188, 150,
242). Both are the open points named above (trainer speed in the first 17
minutes; a model on screen chosen on few validation lesion scans), not
questions about the setting.

**The U-Net on the same sessions, strokes only** (`replay_sparse_unet`,
`replay_sparse_seed1_unet`, `replay_sparse_seed2_unet`, 2026-11-15,
`--model-type unet --supervision sparse`, seeds 0, 1 and 2, commit `09c5426`
with a clean working tree. In all twelve folds no patient is on two sides, no
held-out scan was annotated, every sampler line reads `(adaptive)`, rollback
is off, the batch size is 12 and the folds ran 67 to 71 epochs.)

This is the comparison row for the model: RootPainter's U-Net, trained from
scratch with RootPainter's optimiser (SGD, learning rate 0.01, Nesterov
momentum 0.99), on the strokes alone. The trial seed fixes the patient folds,
the viewing order and the split whatever the model, so each run replays
exactly the sessions of the RETFound-RFA runs of the same seed and the
differences are paired. It is the U-Net inside RetinaPainter's trainer (same
sampler, loss, validation and checkpoint rule as the other rows), not the
unmodified upstream RootPainter trainer: `--trainer-commit` does not apply to
a session replay.

Model on screen after 200 scans (67 minutes), 0.5 cut-off, seed 0 / seed 1 /
seed 2, then the average over the three seeds with its patient-level interval:

| Model and setting | AUROC | Lesion scans flagged (of 103) | Clean scans cleared (of 387) | False blobs per clean scan | Clinician marks found | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|
| U-Net, strokes only | 0.51 / 0.53 / 0.63; mean 0.56 [0.53, 0.59] | 103 / 103 / 102; 99.7% [98.9, 100] | 0 / 6 / 29; 3% [2, 4] | 7.2 / 5.4 / 2.9; mean 5.2 | 66% | 0 / 0 / 0 |
| RETFound-RFA, strokes only | 0.70 / 0.69 / 0.75; mean 0.71 [0.67, 0.76] | 92 / 92 / 82; 86% [81, 91] | 102 / 135 / 182; 36% [32, 40] | 3.6 / 4.5 / 1.5; mean 3.2 | 68% | 0 / 0 / 0 |
| RETFound-RFA, full setting | 0.84 / 0.78 / 0.78; mean 0.80 [0.76, 0.84] | 91 / 87 / 80; 83% [77, 89] | 277 / 241 / 242; 65% [62, 70] | 0.38 / 0.56 / 0.50; mean 0.48 | 78% | 7 / 9 / 8; mean 8 |

Paired differences on the same sessions at 200 scans, seed 0 / seed 1 /
seed 2, then the average over seeds with its patient-level interval:

| Comparison | AUROC | Lesion scans flagged, points | Clean scans cleared, points | False blobs per clean scan | Clinician marks found, points |
|---|---|---|---|---|---|
| RETFound-RFA minus U-Net, both on strokes only | +0.19 / +0.16 / +0.12; mean +0.15 [+0.11, +0.20] | -11 / -11 / -19; mean -14 [-19, -9] | +26 / +33 / +40; mean +33 [+29, +37] | -3.6 / -0.9 / -1.4; mean -2.0 [-2.4, -1.6] | +2 [-5, +8] |
| RETFound-RFA full setting minus U-Net strokes only | +0.33 / +0.25 / +0.15; mean +0.24 [+0.19, +0.29] | -12 / -16 / -21; mean -16 [-23, -10] | +72 / +61 / +55; mean +62 [+59, +67] | -6.8 / -4.8 / -2.4; mean -4.7 [-5.0, -4.4] | +12 [+5, +18] |

On each seed alone, RETFound-RFA minus U-Net on the same strokes: AUROC +0.19
[+0.12, +0.25], +0.16 [+0.08, +0.23] and +0.12 [+0.05, +0.18]; clean scans
cleared +26 points [+20, +33], +33 [+26, +40] and +40 [+34, +45].

Through the session, averaged over the three seeds (AUROC; lesion scans
flagged; clean scans cleared; false blobs per clean scan):

| Scans viewed | U-Net, strokes only | RETFound-RFA, strokes only | RETFound-RFA minus U-Net on strokes, AUROC | Full setting minus U-Net, AUROC |
|---|---|---|---|---|
| 25 | 0.50; 88%; 14%; 15.6 | 0.48; 73%; 25%; 10.1 | -0.02 [-0.06, +0.02] | -0.01 [-0.05, +0.04] |
| 50 | 0.50; 82%; 18%; 10.5 | 0.55; 98%; 2%; 12.7 | +0.05 [+0.01, +0.09] | +0.09 [+0.04, +0.14] |
| 100 | 0.50; 92%; 6%; 7.1 | 0.60; 95%; 6%; 7.6 | +0.10 [+0.05, +0.14] | +0.21 [+0.16, +0.26] |
| 150 | 0.52; 98%; 3%; 4.7 | 0.72; 93%; 21%; 4.8 | +0.20 [+0.14, +0.25] | +0.25 [+0.20, +0.31] |
| 200 | 0.56; 99.7%; 3%; 5.2 | 0.71; 86%; 36%; 3.2 | +0.15 [+0.11, +0.20] | +0.24 [+0.19, +0.29] |

The U-Net by seed (AUROC; lesion scans flagged of 103; clean scans cleared of
387):

| Scans viewed | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 25 | 0.52; 103; 0 | 0.50; 79; 101 | 0.50; 90; 59 |
| 50 | 0.52; 103; 1 | 0.50; 52; 201 | 0.48; 99; 8 |
| 100 | 0.50; 103; 0 | 0.51; 83; 62 | 0.49; 99; 8 |
| 150 | 0.49; 101; 2 | 0.56; 103; 4 | 0.51; 98; 28 |
| 200 | 0.51 [0.46, 0.56]; 103; 0 | 0.53 [0.47, 0.58]; 103; 6 | 0.63 [0.57, 0.68]; 102; 29 |

- **In a one-hour session the U-Net does not separate lesion scans from clean
  scans.** It ranks at chance through 150 scans on every seed and ends at
  0.56; at the 0.5 cut-off it puts a blob on every scan (0 to 29 of 387 clean
  scans cleared, 2.9 to 7.2 false blobs on each). Its "all lesion scans
  flagged" is therefore not detection.
- **It is the model, not the scoring.** The U-Net learns the strokes: its
  training loss falls and it finds 66% of the clinician's marks on held-out
  scans, as many as RETFound-RFA on the same strokes (68%, +2 points
  [-5, +8]). The two models also paint about as many false blobs on a clean
  scan through 150 scans (4.7 and 4.8 at 150 scans; 5.2 and 3.2 at 200). What
  differs is the confidence: 62 to 81 of the 103 lesion scans and 144 to 298
  of the 387 clean scans carry a U-Net blob at the highest confidence step,
  so the blob score cannot order them and the ranking is chance by ties.
- **RETFound-RFA ranks 0.15 better on the same strokes** ([+0.11, +0.20],
  significant on each seed) and clears 33 points more clean scans at 200
  scans. The ranking difference appears at 50 scans and grows (+0.05, +0.10,
  +0.20, +0.15 at 50, 100, 150 and 200 scans).
- **RetinaPainter's full setting against the U-Net starting point**: AUROC
  0.80 against 0.56, 65% against 3% of clean scans cleared, 0.48 against 5.2
  false blobs on a clean scan, and 78% against 66% of the clinician's marks
  found. Of the +0.24 in ranking, +0.15 comes with the model and +0.09 with
  the supervision and sampling.
- **This is a statement about a one-hour session, not about the U-Net.** On
  seed 2 it starts to rank in the last 50 scans (0.51 to 0.63), and an earlier
  U-Net trained for longer on all the strokes at once
  (`combined_sparse_unet_seed0_norollback`) reached AUROC 0.70 to 0.79 by
  largest predicted blob on the 90 validation scans. What the row shows is
  that the U-Net needs more time and annotations than a session of this
  length gives, on data where four scans in five are clean.

**Is the weak start a shortage of training time or of annotations?**
(`replay_sparse_absent_fg50_first50_retfound_rfa`, 2026-11-14: the seed-0
sessions again, stopped after 51 scans, then 45 more minutes of training on
those scans only; `-- --milestones 25,50 --max-scans 51 --tail-minutes 45
--foreground-tile-fraction 0.5`.) The folds had run 6 epochs by 25 scans, 14 to
15 by 51 scans and 61 to 69 by the end.

| Model on screen | Scans viewed | Minutes | Annotated (with lesion mark) | AUROC | Positive flagged | Negative cleared | False blobs per negative scan | Lesion scans with no blob at any cut-off |
|---|---|---|---|---|---|---|---|---|
| Seed-0 session, first run | 26 | 9 | 26 (7) | 0.49 | 100/103 | 5/387 | 3.9 | 0 |
| Same session, this run | 26 | 9 | 26 (7.5) | 0.52 [0.45, 0.58] | 97/103 | 12/387 | 3.8 | 0 |
| Seed-0 session, first run | 51 | 17 | 51 (12.5) | 0.55 | 80/103 | 108/387 | 2.5 | 5 |
| Same session, this run | 51 | 17 | 50 (12.5) | 0.65 [0.59, 0.71] | 85/103 | 115/387 | 3.2 | 7 |
| This run after 45 more minutes, no new scans | 51 | 62 | 50 (12.5) | 0.75 [0.69, 0.81] | 70/103 | 275/387 | 0.52 | 15 |
| Full session, seed 0 | 200 | 67 | 190 (37) | 0.84 | 91/103 | 277/387 | 0.38 | 7 |
| Full session, seed 1 | 200 | 67 | 188 (36) | 0.78 | 87/103 | 241/387 | 0.56 | 9 |

- **Clearing clean scans is a matter of training time.** On the same 51 scans,
  45 more minutes take the model from 115 to 275 of 387 cleared (+41 points
  [+33, +50]) and from 3.2 to 0.5 false blobs per clean scan. That is as many
  clean scans as the full 200-scan sessions clear (-0.5 points [-6, +6] against
  seed 0, +9 points [+2, +16] against seed 1). The blob on nearly every scan in
  the first 17 minutes is the trainer lagging the annotator, not a lack of
  labels.
- **Finding lesion scans is a matter of annotations.** With 12 to 13 scans
  carrying a lesion mark the longer-trained model flags 70 of 103 (-15 points
  [-25, -5] from its own 17-minute model) against 87 to 91 with 36 to 37, and
  15 lesion scans hold no blob at any cut-off.
- **Ranking is mostly time.** AUROC rises 0.10 [+0.04, +0.17] with no new scan;
  the 150 further scans of a full session add 0.02 [-0.04, +0.09] (seed 1) to
  0.09 [+0.03, +0.15] (seed 0).
- **The early model is not repeatable.** The same session run twice differs by
  0.10 AUROC at 50 scans [+0.04, +0.16] (0.03 at 25 scans), although the seed
  is the same: a replay runs in real time, so the epochs do not fall at the
  same moments. Differences between settings in the first 50 scans, including
  the adaptive share's 0.64 against 0.55 above, are inside that.

So no change of supervision or sampling is needed for the first 17 minutes;
more training per session minute is. That is a speed question for the trainer
and is separate from the setting compared here.

This is the setting a painter project now gets when **Use scans marked RIPL
absent as clean scans** is ticked: the project stores
`foreground_tile_fraction: 0.5` and the painter sends it with
`start_training`. The live trainer also takes `--foreground-tile-fraction F`
for projects that send none; a project's own setting wins. The runners here
pass the share the same way, in the instruction: the flag after `--` if given,
otherwise the project's own setting, otherwise adaptive.

## Looking at the scans every model gets wrong

When different architectures, losses and training lengths miss the same
positive scans and flag the same negative ones, the cause is more likely in
the scans or their labels than in the training. `prepare_error_review.py`
finds those scans from existing evaluations and builds a project to step
through in the painter:

```powershell
python prepare_error_review.py --name review_consistent_errors_val `
    --sync-dir D:\RootPainterSync `
    --overlay-project D:\RootPainterSync\projects\<project whose model draws the overlay> `
    --evaluations <evaluation dir> <evaluation dir> ...
```

Each model is held at its own threshold that clears `--target-specificity`
(default 0.9) of the absent scans, so a conservative model is not counted as
missing everything. The project then holds, in this order:

| Group | Scans |
|---|---|
| `missed_positive` | decided present, flagged by at most a third of the models |
| `found_positive` | decided present, flagged by every model (up to `--max-found`), as a reference |
| `flagged_negative` | decided absent, flagged by at least half the models |

Every scan shows the clinician's own strokes where they exist and one model's
overlay. `<project>/review_list.csv` lists the scans with their position,
group and how many models flag them; `error_review.json` records where the
selection came from. File names embed patient IDs, so the list is written to
disk and never printed.

The project is for looking, not training. It has no training annotations,
`auto_start_training_after_reviews` is 0 so completing reviews never starts
training, and it is declared as `unet` so that a running trainer asked to
segment (the painter asks on every step) creates at most a small model and
never loads foundation weights. Existing overlays are not replaced. As in any
project, the painter wants a decision and "Review complete" before moving
forward; those decisions are saved inside the review project only and are a
second read of the same scans.

## Reference mask encodings (read this before touching the masks)

`ground_truth/reviewed_masks/` is **not** uniformly "binary 0/255":

| Source | Files | Encoding |
|---|---:|---|
| `Raphael_Positive_Curated_20261105_v1` | 190 | greyscale, 0 / 255 |
| `Raphael_Updated_20261021_v1` | 300 | painter segmentation RGBA: cyan `(0, 255, 255, 178)` on transparent |

The RGBA files have an empty red channel. Reading "channel 0" turns 14
positive scans into empty masks, and copying one into `annotations/` would
make every lesion pixel a *background* stroke (green = 255). Always decode with
`mask_io.read_binary_mask`.

Most positive reviewed masks are tight (median about 650 px, one component).
A few are not: in the training split two masks cover 40–50% of the scan with
thousands of fragments, and nine exceed 5,000 px. These are accepted output of
an early model, not lesion contours.

## Absent scans as whole-scan background

```powershell
python prepare_offline_project.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name combined_sparse_absent_retfound_rfa_seed0 --model-type retfound_rfa `
    --seed 0 --supervision sparse_absent
```

`sparse_absent` keeps the clinician's strokes on present scans and makes
every pixel of a scan decided *absent* background. It uses only the scan-level
decision, so it needs no reviewed mask and no extra painting, and it brings in
the absent scans that have no strokes at all (134 of 316 in the training split
of the combined development set). Uncertain scans stay unsupervised. A scan
decided absent that carries a red stroke is refused as a contradiction.

It applies to validation as well, so the validation loss then counts false
positives anywhere on an absent scan. Two side effects to keep in mind when
comparing against `sparse` at a fixed time budget: an epoch is
`max(612, 2 x annotated files)` crops, so more annotated scans make each epoch
longer, and validation scores more scans.

Two variants exist for separating the effects:

- `--supervision sparse_absent_val` applies the whole-scan labels to
  validation only. Training is unchanged, so the run isolates what a
  validation signal that sees false positives does to checkpoint selection.
- `run_offline_training.py --loss-type combined_per_tile` computes the loss
  for each crop and averages those, where the default pools every supervised
  pixel of the batch. A fully labelled crop holds tens of thousands of
  supervised pixels against a few hundred on a crop with strokes, and
  otherwise dominates both loss terms. `--foreground-tile-fraction F` fixes
  the share of crops drawn from lesion-bearing scans, which the adaptive
  sampler lowers when many background-only scans are added (84 lesion-bearing
  of 316 annotated scans is 0.266 for the combined development set). Without
  the flag, a project's own `foreground_tile_fraction` is used (the painter
  stores 0.5 in projects that count absent scans as clean scans).
  `prepare_offline_project.py` stores none, so projects it prepares stay
  adaptive unless the flag is given.

## Dense supervision

```powershell
python prepare_offline_project.py --dataset $ds --sync-dir D:\RootPainterSync `
    --name combined_dense_retfound_rfa_seed0 --model-type retfound_rfa --seed 0 `
    --supervision dense
```

Every pixel of each reviewed scan is supervised: reviewed-mask foreground
becomes red and the rest green. A positive mask covering more than 1% of the
scan or split into more than 10 components is treated as unreliable, and that
scan falls back to the clinician's own sparse strokes
(`--dense-max-foreground-fraction`, `--dense-max-components`). The summary
reports how many scans fell back.

Dense supervision changes three things at once relative to sparse: the pixels
supervised per scan, the scans available (absent scans without strokes become
fully supervised negatives), and the validation signal used for checkpoint
selection. Treat a sparse-vs-dense result as a comparison of training policies.
