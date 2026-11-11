"""Create a curated training dataset from a completed positive-candidate annotation session.

Takes a RetinaPainter positive annotation project directory (created by
prepare_positive_annotation_dataset.py) with all reviews completed, and
produces a local curated dataset in the standard ground-truth format.

No network access, model training, or changes to source projects are made.
An existing output directory is never replaced. Clinical records stay on disk.
CLI output contains aggregate counts only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'painter/src/main/python'))
from review_utils import reconstruct_review_binary

from prepare_curated_training_dataset import sha256, write_csv


def _load_project(project: Path):
    configs = list(project.glob('*.seg_proj'))
    if len(configs) != 1:
        raise ValueError('Expected one project configuration file')
    meta_path = project / 'preparation_metadata.json'
    if not meta_path.exists():
        raise ValueError('Expected preparation_metadata.json — is this a positive annotation project?')
    metadata = json.loads(meta_path.read_bytes().decode('utf-8-sig'))
    if metadata.get('final_test_created') is not False:
        raise ValueError('Expected a development session (final_test_created must be false)')
    return metadata


def _load_reviews(project: Path):
    records_path = project / 'reviews/review_records.json'
    if not records_path.exists():
        raise ValueError('reviews/review_records.json not found')
    document = json.loads(records_path.read_bytes().decode('utf-8-sig'))
    if document.get('schema_version') != 1 or not isinstance(document.get('reviews'), dict):
        raise ValueError('Unsupported review schema in review_records.json')
    reviews = document['reviews']
    incomplete = [n for n, r in reviews.items() if not r.get('review_complete')]
    if incomplete:
        raise ValueError(f'{len(incomplete)} incomplete review(s) — complete all reviews before preparing the dataset')
    for name, record in reviews.items():
        if record.get('decision') not in ('present', 'absent', 'uncertain'):
            raise ValueError(f'Invalid review decision for {name}')
    return reviews, records_path


def _load_manifest(project: Path):
    path = project / 'manifest.csv'
    if not path.exists():
        raise ValueError('manifest.csv not found in project folder')
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    required = {'filename', 'patient_id', 'split', 'image_sha256', 'width', 'height'}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f'manifest.csv is missing columns: {missing}')
    return {row['filename']: row for row in rows}


def _load_assignments(project: Path):
    path = project / 'patient_assignments.csv'
    if not path.exists():
        raise ValueError('patient_assignments.csv not found in project folder')
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assignments = {}
    for row in rows:
        pid = row.get('patient_id', '').strip()
        split = row.get('split', '').strip()
        if not pid or split not in ('train', 'val'):
            raise ValueError('Invalid row in patient_assignments.csv')
        if pid in assignments:
            raise ValueError(f'Duplicate patient_id in patient_assignments.csv: {pid}')
        assignments[pid] = split
    return assignments


def _find_annotation(project: Path, filename: str, split: str):
    stem = Path(filename).stem
    path = project / 'annotations' / split / (stem + '.png')
    return path if path.exists() else None


def _read_annotation(path: Path, expected_size):
    with Image.open(path) as img:
        img.load()
        if img.size != expected_size:
            raise ValueError(f'Annotation size {img.size} does not match image size {expected_size}')
        rgba = np.asarray(img.convert('RGBA'))
    red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
    green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
    return red, green


def _snapshot(project: Path, record: dict, kind: str, watched: dict, expected_size):
    relative = record.get(kind + '_snapshot')
    if not relative:
        if kind == 'annotation':
            return None
        raise ValueError(f'Required {kind} snapshot is missing')
    path = project / relative
    if (Path(relative).is_absolute()
            or not path.resolve().is_relative_to(project.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError(f'{kind} snapshot path is not confined to the project')
    digest = sha256(path)
    if digest != record.get(kind + '_sha256'):
        raise ValueError(f'{kind} snapshot hash mismatch')
    with Image.open(path) as img:
        img.load()
        if img.size != expected_size:
            raise ValueError(f'{kind} snapshot dimensions mismatch')
    watched[path] = digest
    return path


def _png(mask, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path, format='PNG')


def prepare_positive_curated_dataset(project_dir, images_dir, output_dir):
    project = Path(project_dir).resolve()
    images = Path(images_dir).resolve()
    output = Path(output_dir).resolve()

    if output.exists():
        raise ValueError('Output directory already exists; refusing to overwrite')
    if output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError('Output cannot contain or be contained by the source project')
    if not images.is_dir():
        raise ValueError(f'Images directory not found: {images}')

    metadata = _load_project(project)
    reviews, records_path = _load_reviews(project)
    manifest = _load_manifest(project)
    assignments = _load_assignments(project)

    # Verify every review has a manifest entry
    missing_manifest = [n for n in reviews if n not in manifest]
    if missing_manifest:
        raise ValueError(f'{len(missing_manifest)} review(s) have no manifest entry')

    watched = {records_path: sha256(records_path)}

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()

    gt = stage / 'ground_truth'
    for folder in ('clinician_foreground', 'known_pixels', 'reviewed_masks'):
        (gt / folder).mkdir(parents=True)

    labels, sparse_index, reviewed_index = [], [], []
    image_count = 0
    eligible_count = 0
    reviewed_count = 0

    for filename, record in reviews.items():
        mrow = manifest[filename]
        patient_id = mrow['patient_id']
        split = assignments.get(patient_id)
        if split is None:
            raise ValueError(f'No patient assignment for patient {patient_id} ({filename})')
        if mrow['split'] != split:
            raise ValueError(f'Manifest split {mrow["split"]} conflicts with assignment {split} for {filename}')

        decision = record['decision']
        expected_size = (int(mrow['width']), int(mrow['height']))

        # Copy image
        src_image = images / filename
        if not src_image.exists():
            raise ValueError(f'Source image not found: {src_image}')
        img_key = 'images/' + filename
        dest_image = stage / img_key
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, dest_image)
        actual_sha = sha256(dest_image)
        if actual_sha != mrow['image_sha256']:
            raise ValueError(f'Image hash mismatch for {filename}')
        image_count += 1

        # Scan label
        definite = decision in ('present', 'absent')
        labels.append(dict(
            filename=filename,
            patient_id=patient_id,
            split=split,
            decision=decision,
            binary_label=int(decision == 'present') if definite else '',
            binary_evaluation_eligible=definite,
            reference_status=mrow.get('reference_status', ''),
            independently_adjudicated=False,
        ))

        # Find annotation
        annot_path = _find_annotation(project, filename, split)
        eligible = False
        red_pixels = 0
        green_pixels = 0
        annot_sha = ''
        exclusion_reason = ''

        if decision == 'uncertain':
            exclusion_reason = 'uncertain_decision'
        elif annot_path is None:
            exclusion_reason = 'no_annotation_file'
        else:
            red, green = _read_annotation(annot_path, expected_size)
            if not (red | green).any():
                exclusion_reason = 'annotation_has_no_pixels'
            elif decision == 'absent' and red.any():
                exclusion_reason = 'absent_annotation_has_red_pixels'
            else:
                eligible = True
                red_pixels = int(red.sum())
                green_pixels = int(green.sum())
                annot_sha = sha256(annot_path)
                watched[annot_path] = annot_sha

                # Copy annotation
                annot_name = Path(filename).stem + '.png'
                dest_annot = stage / 'annotations' / split / annot_name
                dest_annot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(annot_path, dest_annot)
                if sha256(dest_annot) != annot_sha:
                    raise ValueError(f'Annotation copy hash mismatch for {filename}')

                # Ground-truth masks
                _png(red, gt / 'clinician_foreground' / annot_name)
                _png(red | green, gt / 'known_pixels' / annot_name)
                eligible_count += 1

        sparse_index.append(dict(
            filename=filename,
            split=split,
            decision=decision,
            reference_type='sparse_corrective_annotation',
            exhaustive_lesion_reference=False,
            training_eligible=eligible,
            foreground_mask='clinician_foreground/' + Path(filename).stem + '.png' if eligible else '',
            known_pixel_mask='known_pixels/' + Path(filename).stem + '.png' if eligible else '',
            red_pixels=red_pixels,
            green_pixels=green_pixels,
            annotation_sha256=annot_sha,
            status=exclusion_reason or 'available_sparse_only',
        ))

        # Reviewed mask
        reviewed = dict(
            filename=filename,
            split=split,
            decision=decision,
            mask='',
            reference_type='completed_review_reconstruction',
            independently_adjudicated=False,
            exhaustive_lesion_reference=False,
            reconstruction_mode='',
            segmentation_sha256='',
            annotation_sha256='',
            status='no_completed_review_source',
        )
        if definite:
            try:
                seg_path = _snapshot(project, record, 'segmentation', watched, expected_size)
                ann_path = _snapshot(project, record, 'annotation', watched, expected_size)
                mode = record.get('reconstruction_mode', 'prediction_corrected')
                mask = reconstruct_review_binary(seg_path, ann_path, decision=decision, reconstruction_mode=mode)
                reviewed['reconstruction_mode'] = mode
                reviewed['segmentation_sha256'] = record['segmentation_sha256']
                reviewed['annotation_sha256'] = record.get('annotation_sha256') or ''
                if decision == 'present' and not mask.any():
                    reviewed['status'] = 'positive_review_has_empty_mask_needs_correction'
                else:
                    rel = 'reviewed_masks/' + Path(filename).stem + '.png'
                    _png(mask, gt / rel)
                    reviewed['mask'] = rel
                    reviewed['status'] = 'available_completed_review'
                    reviewed_count += 1
            except (ValueError, FileNotFoundError) as err:
                reviewed['status'] = 'reconstruction_failed'
                reviewed['mask'] = ''
        reviewed_index.append(reviewed)

    for csv_name, entries in (
            ('bscan_labels.csv', labels),
            ('sparse_mask_index.csv', sparse_index),
            ('reviewed_mask_index.csv', reviewed_index)):
        write_csv(gt / csv_name, entries, list(entries[0]))

    # Verify source hasn't changed
    for path, expected in watched.items():
        if sha256(path) != expected:
            raise ValueError('Source changed during preparation; staging was not published')

    decisions = dict(Counter(r['decision'] for r in reviews.values()))
    result = dict(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='positive_candidate_curated_dataset',
        final_test_created=False,
        images=image_count,
        decisions=decisions,
        eligible_sparse_annotations=eligible_count,
        reviewed_masks=reviewed_count,
        reviewed_mask_statuses=dict(Counter(r['status'] for r in reviewed_index)),
        splits={split: dict(
            images=sum(r['split'] == split for r in labels),
            patients=len({r['patient_id'] for r in labels if r['split'] == split}))
            for split in ('train', 'val')},
        source_project=str(project),
        source_images=str(images),
        source_preparation_metadata=metadata.get('created_at_utc', ''),
        model_type=metadata.get('model_type', ''),
        seed=metadata.get('seed'),
    )

    readme = (
        'RIPL POSITIVE-CANDIDATE CURATED DATASET\n\n'
        'Images and reference files are local. No model was trained by this export.\n'
        'Use images/ as the image directory in RetinaPainter. '
        'Patient assignments are in manifest.csv and patient_assignments.csv.\n\n'
        'ground_truth/bscan_labels.csv: scan-level present/absent/uncertain decisions.\n'
        'ground_truth/clinician_foreground/: binary 0/255 foreground masks (red pixels).\n'
        'ground_truth/known_pixels/: binary 0/255 known-pixel masks (red | green).\n'
        'ground_truth/reviewed_masks/: masks reconstructed from completed review snapshots.\n'
        'ground_truth/sparse_mask_index.csv: annotation availability and eligibility.\n'
        'ground_truth/reviewed_mask_index.csv: reviewed mask availability.\n\n'
        'Positive-only session: absent scans have no red marks. '
        'Uncertain decisions are excluded from training supervision.\n'
        'This is a development dataset, not a final test cohort.\n\n'
        'AGGREGATE SUMMARY\n'
    )
    (stage / 'CURATED_DATASET_README.txt').write_text(
        readme + json.dumps(result, indent=2) + '\n', encoding='utf-8')
    result['file_sha256'] = {
        p.relative_to(stage).as_posix(): sha256(p)
        for p in sorted(stage.rglob('*')) if p.is_file()
        and p.name != 'curated_dataset_metadata.json'
    }
    (stage / 'curated_dataset_metadata.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')

    if output.exists():
        raise ValueError('Output appeared during preparation; refusing replacement')
    stage.rename(output)
    return {k: v for k, v in result.items() if k != 'file_sha256'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path,
                        help='Path to the completed positive annotation project folder')
    parser.add_argument('--images', type=Path,
                        help='Path to the source images folder (default: curated_datasets/<project_name>/images)')
    parser.add_argument('--output', required=True, type=Path,
                        help='Path for the new curated dataset (must not exist)')
    args = parser.parse_args()

    project = args.project.resolve()
    if args.images:
        images = args.images.resolve()
    else:
        # Convention: curated_datasets sibling of projects, same name
        images = project.parent.parent / 'curated_datasets' / project.name / 'images'

    try:
        result = prepare_positive_curated_dataset(project, images, args.output)
    except Exception as error:
        parser.exit(2, 'Dataset was not published (' + type(error).__name__ + '). '
                    'Inspect sources locally; no clinical details were printed.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
