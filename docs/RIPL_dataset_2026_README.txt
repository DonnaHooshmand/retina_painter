RIPL_dataset_2026 nested clinician-annotation subsets

Source: D:\RootPainterSync\datasets\SD-OCT_RIPL_AI_Database
Seed: 2026
Ranking algorithm: sha256_seeded_filename_rank_v1
Subset sizes: 150, 200, 250, 300, 350, 400, 450, 500, 550, 600
Excluded directories: D:\RootPainterSync\datasets\Jon_6-11
Excluded matching source images: 750

The seed and each filename are hashed to create one reproducible random order.
Folder N contains ranks 1 through N. Therefore every smaller folder is an
exact subset of every larger folder. Images are independent copies, not links.

Folder contents and intended use:
- 150 contains manifest ranks 1-150.
- 200 contains ranks 1-200, including every image in 150.
- Each later folder adds the next 50 ranks through 600.
- The folders support nested sample-size experiments. They are not ten
  independent random samples.

For clinician collection, create one RetinaPainter project per doctor using
the 600 folder. Annotate each scan once. Use manifest rank later to derive the
150, 200, ..., 550 analysis subsets from those same reviews; do not ask a
doctor to annotate every numeric folder separately. Keep annotations and
exports in the RetinaPainter project directory and do not edit these dataset
image copies.

Clinician review protocol:
- For each B-scan select RIPL present, RIPL absent, or uncertain, then check
  Review complete only after inspecting the entire scan.
- Before Network > Start training is selected for the first time, ignore the
  random model prediction and paint every true RIPL red. A confirmed absent
  scan needs no exhaustive green cleanup; its completed export is empty by
  definition.
- A pre-training present review cannot be completed without red foreground,
  and an absent review cannot be completed while red marks remain.
- Training starts automatically after the tenth completed review is saved and
  you advance. Forward navigation requires a completed decision. If an early
  review was missed, RetinaPainter names that image and keeps you in the
  initial set until it is completed. You can still use Network > Start
  training earlier for a deliberately different protocol.
- After training starts, red marks missed RIPLs, green marks false-positive
  predictions, and correct model suggestions may be left untouched.
- A completed review saves a separate labeled audit image under
  reviews/decision_previews. The live UI stays uncluttered, and the label is
  never written into the source OCT or sparse correction PNG or given to the
  model.
- Uncertain reviews remain uncertain and must not be counted as positive or
  negative outcomes without a separate adjudication rule.

RetinaPainter detects this parent manifest when a project is created from a
numeric child folder and uses manifest rank as its navigation order. Thus the
first 150 images of every larger trial exactly match the ordered 150 trial.
The project trial seed still controls model and training randomness.

manifest.csv records rank, the first subset containing each image, patient ID,
eye, byte size, hash, and original source path. dataset_metadata.json records
the construction settings. Keep both files with the collection because they
freeze selection and provenance.

The selection is image-level, not patient-grouped. Before research model
evaluation, assign train/validation/test partitions by patient_id from the
manifest so images from one patient cannot cross evaluation boundaries. This
collection is not the final held-out test set.
