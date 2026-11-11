"""Create a painter-compatible project pre-populated for offline training.

The project is built from a curated dataset in the standard layout
(``images/``, ``annotations/train|val/``, ``ground_truth/*.csv``). Nothing is
trained here and the dataset is never modified. The resulting project can be
trained with ``run_offline_training.py``, scored with ``evaluate_project.py``
and opened in the painter to inspect predictions.

Two supervision policies are supported so they can be compared deliberately:

``sparse`` (default)
    Copy the clinician's red/green corrective strokes exactly as exported.
    Untouched pixels stay unknown and contribute no loss.

``dense``
    Supervise every pixel of each reviewed scan: reviewed-mask foreground
    becomes red, everything else green. A reviewed mask is the clinician's
    accepted *corrected prediction*, so it can contain uncorrected output of
    an earlier model. A positive mask that is implausibly large or fragmented
    is therefore not trusted as a dense label; that scan falls back to its
    sparse strokes (or is left out when it has none). This is a training-policy
    change relative to sparse corrective annotation, not a free upgrade.

Train/validation membership always follows the dataset's patient-separated
``split`` column. CLI output is aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from mask_io import read_annotation, read_binary_mask, write_annotation

MODEL_TYPES = ('unet', 'retfound', 'retfound_rfa')
PROJECT_SUBFOLDERS = ('annotations/train', 'annotations/val', 'segmentations',
                      'models', 'messages', 'logs', 'reviews',
                      'review_exports')


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def seeded_file_order(fnames, seed):
    """Match the painter's ``project_order.seeded_file_order``."""
    ordered = sorted(fnames)
    random.Random(seed).shuffle(ordered)
    return ordered


def check_patient_separation(label_rows):
    """Refuse a dataset whose patients appear in more than one split."""
    patient_splits = {}
    for row in label_rows:
        patient = row.get('patient_id') or ''
        if not patient or patient == 'unknown':
            continue
        if patient_splits.setdefault(patient, row['split']) != row['split']:
            raise ValueError(
                'Dataset is not patient-separated: at least one patient '
                'occurs in both train and val')


def reference_is_reliable(mask, max_foreground_fraction, max_components):
    """Return whether a positive reviewed mask is plausible as a dense label."""
    if not mask.any():
        return False
    if mask.mean() > max_foreground_fraction:
        return False
    from skimage.measure import label
    return int(label(mask, connectivity=2).max()) <= max_components


def populate_sparse(dataset, project, label_rows):
    counts = Counter()
    splits = {row['filename']: row['split'] for row in label_rows}
    stems = {Path(name).stem: split for name, split in splits.items()}
    for split in ('train', 'val'):
        for source in sorted((dataset / 'annotations' / split).glob('*.png')):
            if stems.get(source.stem) != split:
                raise ValueError(
                    'An annotation is stored under a split that disagrees '
                    'with bscan_labels.csv')
            foreground, background = read_annotation(source)
            shutil.copy2(source, project / 'annotations' / split / source.name)
            counts[f'{split}_files'] += 1
            counts[f'{split}_foreground_files'] += int(foreground.any())
            counts[f'{split}_background_files'] += int(background.any())
    return counts


def populate_dense(dataset, project, max_foreground_fraction, max_components):
    counts = Counter()
    ground_truth = dataset / 'ground_truth'
    for row in read_rows(ground_truth / 'reviewed_mask_index.csv'):
        split = row['split']
        if row['status'] != 'available' or row['decision'] not in (
                'present', 'absent'):
            counts[f'{split}_without_reviewed_mask'] += 1
            continue
        stem = Path(row['filename']).stem
        mask = read_binary_mask(ground_truth / row['mask'])
        destination = project / 'annotations' / split / (stem + '.png')
        if row['decision'] == 'absent':
            if mask.any():
                raise ValueError(
                    'A scan decided absent has reviewed-mask foreground')
            dense = True
        else:
            dense = reference_is_reliable(
                mask, max_foreground_fraction, max_components)
        if dense:
            write_annotation(destination, mask, ~mask)
            counts[f'{split}_dense_files'] += 1
            counts[f'{split}_foreground_files'] += int(mask.any())
            continue
        sparse = dataset / 'annotations' / split / (stem + '.png')
        if sparse.is_file():
            shutil.copy2(sparse, destination)
            counts[f'{split}_sparse_fallback_files'] += 1
            counts[f'{split}_foreground_files'] += int(
                read_annotation(sparse)[0].any())
        else:
            counts[f'{split}_unreliable_mask_skipped'] += 1
    return counts


def prepare_offline_project(dataset, sync_dir, name, model_type='retfound_rfa',
                            seed=0, supervision='sparse',
                            dense_max_foreground_fraction=0.01,
                            dense_max_components=10):
    dataset = Path(dataset).resolve()
    sync_dir = Path(sync_dir).resolve()
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unknown model type: {model_type}')
    if supervision not in ('sparse', 'dense'):
        raise ValueError(f'Unknown supervision policy: {supervision}')
    images_dir = dataset / 'images'
    if not images_dir.is_dir():
        raise ValueError(f'Dataset has no images directory: {images_dir}')

    label_rows = read_rows(dataset / 'ground_truth' / 'bscan_labels.csv')
    check_patient_separation(label_rows)
    image_names = {path.name for path in images_dir.iterdir()
                   if path.is_file()}
    missing = [row['filename'] for row in label_rows
               if row['filename'] not in image_names]
    if missing:
        raise ValueError(f'{len(missing)} labelled image(s) are missing')

    project = sync_dir / 'projects' / name
    if project.exists():
        raise ValueError(
            f'Project already exists; refusing to overwrite: {project}')
    for folder in PROJECT_SUBFOLDERS:
        (project / folder).mkdir(parents=True)

    if supervision == 'sparse':
        counts = populate_sparse(dataset, project, label_rows)
    else:
        counts = populate_dense(dataset, project,
                                dense_max_foreground_fraction,
                                dense_max_components)

    ordered = seeded_file_order(
        [row['filename'] for row in label_rows], seed)
    split_of = {row['filename']: row['split'] for row in label_rows}
    dataset_rel = os.path.relpath(images_dir, sync_dir / 'datasets')
    project_info = {
        'name': name,
        'dataset': Path(dataset_rel).as_posix(),
        'original_model_file': 'random weights',
        'location': str(PurePosixPath('projects') / name),
        'file_names': ordered,
        'image_order_method': 'seeded_shuffle_v1',
        'dataset_manifest': None,
        'image_order_seed': seed,
        'training_seed': seed,
        'train_file_names': [f for f in ordered if split_of[f] == 'train'],
        'val_file_names': [f for f in ordered if split_of[f] == 'val'],
        'split_method': 'dataset_patient_split_v1',
        'model_type': model_type,
        'review_schema_version': 1,
        'auto_start_training_after_reviews': 10,
    }
    with open(project / (name + '.seg_proj'), 'w', encoding='utf-8') as stream:
        json.dump(project_info, stream, indent=4)

    summary = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(project),
        'dataset': str(dataset),
        'images_dir': str(images_dir),
        'model_type': model_type,
        'seed': seed,
        'supervision': supervision,
        'images': len(label_rows),
        'annotation_counts': dict(sorted(counts.items())),
    }
    if supervision == 'dense':
        summary['dense_max_foreground_fraction'] = dense_max_foreground_fraction
        summary['dense_max_components'] = dense_max_components
    with open(project / 'offline_experiment.json', 'w',
              encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, type=Path,
                        help='curated dataset directory in the standard layout')
    parser.add_argument('--sync-dir', required=True, type=Path,
                        help='RetinaPainter sync directory')
    parser.add_argument('--name', required=True, help='new project name')
    parser.add_argument('--model-type', default='retfound_rfa',
                        choices=MODEL_TYPES)
    parser.add_argument('--seed', type=int, default=0, help='trial seed')
    parser.add_argument('--supervision', default='sparse',
                        choices=('sparse', 'dense'))
    parser.add_argument(
        '--dense-max-foreground-fraction', type=float, default=0.01,
        help='dense only: a positive reviewed mask covering more of the scan '
             'than this is treated as unreliable (default 0.01)')
    parser.add_argument(
        '--dense-max-components', type=int, default=10,
        help='dense only: a positive reviewed mask with more connected '
             'components than this is treated as unreliable (default 10)')
    args = parser.parse_args()
    summary = prepare_offline_project(
        args.dataset, args.sync_dir, args.name, model_type=args.model_type,
        seed=args.seed, supervision=args.supervision,
        dense_max_foreground_fraction=args.dense_max_foreground_fraction,
        dense_max_components=args.dense_max_components)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
