# RIPL_dataset_2026 construction

`RIPL_dataset_2026` is a reproducible nested collection for prospective
clinician annotation. It is separate from the small 2026 engineering trial
dataset.

## Planned layout

```text
D:\RootPainterSync\datasets\RIPL_dataset_2026\
  150\
  200\
  250\
  300\
  350\
  400\
  450\
  500\
  550\
  600\
  manifest.csv
  dataset_metadata.json
  README.txt
```

The source is `D:\RootPainterSync\datasets\SD-OCT_RIPL_AI_Database`.
Exact image filenames already present in `D:\RootPainterSync\datasets\Jon_6-11`
are excluded to avoid asking clinicians to repeat the earlier annotation
collection. Patients are not excluded because that prior collection spans 160
of the source database's 187 patient identifiers and patient-level exclusion
would leave a narrow 27-patient pool.

The selection seed is `2026`. The script computes a SHA-256 rank from the seed
and filename and selects the first 600 ranks. Folder `N` contains ranks 1
through `N`, guaranteeing that all 150 images occur in the 200-image folder,
all 200 occur in the 250-image folder, and so on. The root manifest freezes the
exact order, file hashes, patient IDs, and provenance.

The sampling is image-level. It must not be treated as a research train/test
split. After annotation, model-development and held-out evaluation partitions
must be assigned by patient ID so B-scans from one patient cannot cross those
boundaries.

Recreate or verify the collection with:

```powershell
python prepare_nested_annotation_datasets.py `
  --source D:\RootPainterSync\datasets\SD-OCT_RIPL_AI_Database `
  --output D:\RootPainterSync\datasets\RIPL_dataset_2026 `
  --exclude-dir D:\RootPainterSync\datasets\Jon_6-11 `
  --seed 2026
```

For clinician collection, create one separate RetinaPainter project per doctor
and select the `600` child folder, not the `RIPL_dataset_2026` parent. Each
doctor reviews those 600 images once. The smaller sample sizes are derived
later using `manifest.csv`: folder `150` is ranks 1–150, folder `200` is ranks
1–200, and each later folder adds the next 50 ranks. Doctors should not
annotate all ten folders separately.

The numeric folders support nested sample-size experiments; they are not ten
independent random samples. Keep annotations, review records, and exported
masks in the RetinaPainter project rather than editing these dataset copies.
The manifest records each image's rank, first containing subset, patient ID,
eye, size, hash, and source path. `dataset_metadata.json` records the selection
algorithm and construction inputs.

For each scan the clinician selects `RIPL present`, `RIPL absent`, or
`uncertain`, then marks `Review complete`. Before training first starts, the
random prediction is ignored for mask export and the clinician must paint
every true RIPL red. A confirmed absent review exports an empty mask without
requiring exhaustive green coverage of random false positives. After training
starts, the usual corrective workflow applies: red adds missed RIPLs, green
removes false positives, and correct suggestions can remain untouched. See
`docs/clinician_review_workflow.md` for the complete provenance and export
contract. New projects automatically start training after the tenth completed
review is saved and the clinician advances. Forward navigation requires the
current review to be complete, and the warm-up gate identifies any missing
review among the first ten. Manual start remains available for deliberately
different protocols.

When a project is created from any numeric folder, RetinaPainter detects the
parent manifest and uses its rank as the navigation order instead of applying
a separate shuffle to each folder. Consequently, the first 150 scans in the
250-image trial are exactly the same ordered scans as the complete 150-image
trial. The project trial seed still controls model initialization, training
sampling, worker randomness, and other stochastic training behavior.
