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
validation, checkpoint promotion, automatic rollback and early stopping are the
painter workflow's own code. Three things differ, all transport:

- The shared `<sync>/instructions` queue is neither cleared at start-up nor
  polled, so an open painter cannot switch the model mid-run and several runs
  can train in parallel.
- Event logs go to `<project>/logs/offline_server_log.txt` instead of the
  shared `server_log.txt`; painter status messages are not written.
- `--gpu` pins the run to one device.

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
