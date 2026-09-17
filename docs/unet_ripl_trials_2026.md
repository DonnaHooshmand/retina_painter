# Interactive U-Net RIPL trials, 2026

These trials are engineering smoke tests of whether RetinaPainter can learn
small RIPL-like targets from sparse corrective annotation. They are not a
clinical performance study. The reference annotations are incomplete and the
reported blob counts were recorded during interaction rather than produced by
a blinded held-out evaluator.

## Trial 5

- Project: `donna_UNET_seed0_retry5_3007`
- Model: scratch-trained U-Net using random weights and trial seed 0.
- Training began after seven images had been saved.
- The fixed sampler drew 50% of crops from the small foreground-bearing pool.
  At the observed annotation distribution, this sampled each foreground file
  roughly 3 to 3.6 times as often as each background file.
- Across matched images 15 through 30, approximately 35 false-positive blobs
  were reported. There were also missed lesions and predictions outside the
  retinal layers.
- Saved-checkpoint replay showed no hidden strong candidate: later checkpoints
  traded reduced false-positive burden for poor foreground recall. The UI did
  advance, so stale checkpoint routing was not the primary cause.
- Resulting change: replace the fixed 50% target with an adaptive correction-
  pool target while leaving the architecture, loss, seed, checkpoint logic,
  and 612 crops per epoch unchanged.

## Trial 6

- Project: `donna_UNET_seed0_1109`
- Dataset: `curated_images_for_testing` (318 images).
- Model: scratch-trained U-Net using random weights and trial seed 0.
- Training began after seven images had been saved.
- The adaptive sampler operated as intended. Logged foreground targets ranged
  from 18% to 33% and ended at 22% for 9 foreground-bearing and 32
  background-bearing training annotations.
- Manual observations for images 15 through 56 recorded approximately 73
  false-positive blobs, 5 correctly detected RIPL objects, and 11 missed RIPL
  objects. Twelve scans were reported as clean negatives; image 54 had only
  sub-blob noise, and image 32 had a doctor disagreement.
- In the matched image 15 through 30 interval, Trial 6 had approximately 34
  false-positive blobs versus approximately 35 in Trial 5. Adaptive sampling
  therefore did not materially resolve the false-positive problem.

### Saved annotation evidence

| Split | Files | Foreground-bearing | Background-bearing | Red pixels | Green pixels |
|---|---:|---:|---:|---:|---:|
| Train | 34 | 9 | 32 | 13,559 | 693,276 |
| Validation | 9 | 1 | 9 | 1,332 | 173,592 |
| Total | 43 | 10 | 41 | 14,891 | 866,868 |

Files can occur in both bearing counts when they contain both correction
colors. Pixel counts describe sparse brush supervision, not dense lesion
prevalence.

### Checkpoint evidence

- Validation contained no foreground until 16:44. Before that point,
  checkpoints 2 through 21 were promoted provisionally using background-only
  cross-entropy. Checkpoint 21 reached a background validation loss near
  `0.000002`, which strongly rewards suppressing predictions but provides no
  evidence of RIPL sensitivity.
- Checkpoints 22 and 23 were promoted after the one foreground-bearing
  validation annotation arrived. At 16:47:20, checkpoint 23 improved masked
  validation loss from `1.026861` to `0.876826`; its diagnostic hard pixel F1
  was `0.1586`.
- Checkpoint 23 never advanced afterward. Twelve logged candidate rollbacks
  restored it after three consecutive worse foreground-informed validation
  epochs. Later annotation changes re-evaluated the same checkpoint and its
  diagnostic F1 fell to `0.1226` and then `0.0727` as validation supervision
  expanded.
- Training stopped cleanly at 17:14:33. The UI was intentionally using the
  retained checkpoint rather than accidentally loading an unrelated or stale
  project.

### Diagnosis

The adaptive sampler fixed the specific Trial 5 oversampling mechanism, but it
was not the dominant remaining limitation. Trial 6's fixed seeded split was
created before labels were known, and rare positives left validation without
foreground for most of the interactive session. Background-only promotion
could assess false-positive suppression but not sensitivity. Once a positive
entered validation, one foreground-bearing validation image was too little
diversity to select a model for B-scan-level RIPL behavior. Repeated rollback
then prevented later candidates from replacing checkpoint 23 unless they
improved that narrow sparse pixel objective within three epochs.

The optimization and selection surrogate also differs from the intended
clinical endpoint: masked pixel Dice plus cross-entropy on sparse strokes does
not directly minimize false-positive objects per scan or maximize B-scan
sensitivity. The trial therefore demonstrated occasional useful recognition,
but not reliable small-data RIPL detection.

### Implemented follow-up

Do not interpret Trial 6 as evidence that the adaptive sampler is broken.
Automatic candidate rollback now requires at least two independently
foreground-bearing validation annotation files. With exactly one, checkpoint
promotion and training continue but rollback is explicitly deferred and the
terminal reports `1/2 foreground-bearing files`. The three-epoch rollback
patience begins only after the minimum evidence exists. This is the smallest
guard against the specific Trial 6 failure and does not make a two-image
validation set statistically strong. Official clinician annotation should use
the larger reproducibly sampled collection, followed by patient-separated
evaluation.
