# Clinician review workflow

RetinaPainter records the clinician's B-scan-level decision separately from
the sparse corrective annotation. Each displayed scan has four states:

- no decision selected;
- `RIPL present`;
- `RIPL absent`; or
- `Uncertain`.

`Review complete` is enabled only after a decision is selected. The clinician
should check it only after inspecting the full B-scan and making all required
corrections. A scan can therefore have no correction PNG and still be an
explicitly reviewed negative. Conversely, opening or partially correcting a
scan does not make it complete.

Corrections retain the RootPainter meaning:

- red adds missed foreground;
- green removes false-positive foreground; and
- after training begins, untouched pixels retain the displayed model
  prediction.

Before **Network > Start training** is selected for the first time, the
displayed random-weight prediction is not treated as label evidence. Initial
seed reviews export only clinician-painted red foreground; green marks may
still be saved as sparse training corrections but do not need to exhaustively
cover random false positives. Starting training writes a persistent project
marker, so stopping or reopening the painter does not return the project to
this initial mode. During this initial phase the clinician must mark every RIPL
red, even if a random cyan prediction happens to overlap it. RetinaPainter will
not complete a pre-training `RIPL present` review unless it contains red
foreground, and it will not complete an `RIPL absent` review that contains red
foreground.

New projects automatically start training after the tenth completed review is
saved and the clinician advances to another image. Ten is a human-readable
warm-up target and provides eight training-routed and two validation-routed
scans under the fixed 5:1 filename split. It is not a statistical claim that
ten scans guarantee class balance or foreground-bearing validation. Manual
**Network > Start training** remains available when an intentional protocol
uses a different warm-up size. Forward navigation requires the current scan
to be review-complete. At the end of the warm-up, RetinaPainter also verifies
that all first ten scans are complete and identifies the first missing review
instead of silently postponing automatic training. Previous navigation remains
available so an incomplete scan can be corrected.

Saving a completed review records the decision in
`reviews/review_records.json` and snapshots the displayed prediction and
correction image under `reviews/`. These snapshots are hashed and are the
inputs to export. Later live-model activity therefore cannot silently change
the completed label. RetinaPainter also writes a labeled visual copy under
`reviews/decision_previews/` for human sanity checking. The label appears only
in that audit preview—not in the live UI, the source OCT, the sparse correction
PNG, or any image supplied to the model.

Use **Project > Export completed reviews** to create a timestamped directory
under `review_exports/`. It contains:

- `reviews.csv` with the scan decisions, completion timestamps, hashes,
  foreground-pixel counts, and decision/mask consistency checks;
- `binary_masks/` with the reconstructed masks;
- `audit_previews/` with the labeled visual review copies, when available; and
- `export_metadata.json` describing the export and any consistency warnings.

Each binary mask records one of three explicit reconstruction modes:

- `decision_override_empty`: an explicit `RIPL absent` decision exports an
  empty mask, regardless of random or residual predicted pixels;
- `annotation_foreground_only`: before training first starts, the random
  prediction is ignored and only red clinician foreground is exported; or
- `prediction_corrected`: after training starts, the mask is reconstructed as
  displayed prediction - green corrections + red corrections.

A completed `RIPL present` scan with an empty exported mask is reported as a
consistency warning. Uncertain decisions remain explicit and should not be
used as definitive positive or negative labels. The CSV identifies the
reconstruction mode and whether a prediction contributed to every mask.
