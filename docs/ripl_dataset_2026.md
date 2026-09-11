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

When creating a RetinaPainter project, select the desired numeric child folder
instead of the `RIPL_dataset_2026` parent.
