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
  of 316 annotated scans is 0.266 for the combined development set).

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
