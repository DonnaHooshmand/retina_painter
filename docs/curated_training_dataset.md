# Curated development dataset

This describes the earlier frozen replay export. It is not the fresh annotation
practice dataset requested by the user. For that workflow use
`D:\RootPainterSync\curated_datasets\Raphael_091726_practice_20260925` and its
`README.txt`: clean scans, corresponding doctor reference pictures, and a blank
practice project preserving patient partitions.

The local dataset is
`D:\RootPainterSync\datasets\curated\_datasets\RIPL_Development_20260925_v1`.
It contains 613 images, preserving the frozen replay's 495 training and 118
validation images. The 135 training patients and 34 validation patients do not
overlap. It is a development dataset, not an untouched final test cohort.

`images/` contains the scans. `manifest.csv` and `patient_assignments.csv` preserve
the authoritative patient routing. Any training project must honor those roles;
do not recreate the split by randomly dividing individual images.

`ground_truth/bscan_labels.csv` contains 12 present, 256 absent, 30 uncertain,
and 315 unreviewed decisions. Uncertain and unreviewed scans have no definite
binary label. These references are not independently adjudicated.

`ground_truth/reviewed_masks/` contains 267 reconstructed completed-review masks:
11 positive and 256 negative. One additional positive decision has an empty
reconstruction, so its mask is withheld and flagged in `reviewed_mask_index.csv`.
Historical masks can include accepted model predictions; they are not certified
exhaustive lesion contours.

`ground_truth/clinician_foreground/` and `ground_truth/known_pixels/` contain 422
paired binary masks derived from eligible red/green corrections. Pixels outside
the known-pixel mask are unknown, not background. The original sparse inputs
remain under `annotations/train` and `annotations/val`. Uncertain corrections
are withheld. No dense training policy is introduced by this export.

The dataset retains original replay provenance and source file hashes. Its
`curated_dataset_metadata.json` records hashes of every other exported file.
The self-contained exporter refuses existing destinations and checks patient
routing, image content separation, correction semantics, source review version,
snapshot hashes, and copy integrity before publishing a new directory.

Raphael's original 300 completed reviews included 12 positives (4%). The follow-up
project had no returned completed reviews or annotations available locally when
this version was prepared; this export does not invent follow-up labels.

Preparation and verification run locally. Clinical images and patient-level
tables must not be printed into assistant output or uploaded to cloud services.
