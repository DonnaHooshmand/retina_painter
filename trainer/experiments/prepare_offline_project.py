"""Create a painter-compatible project pre-populated for offline training.

The project is built from a curated dataset in the standard layout
(``images/``, ``annotations/train|val/``, ``ground_truth/*.csv``). Nothing is
trained here and the dataset is never modified. The resulting project can be
trained with ``run_offline_training.py``, scored with ``evaluate_project.py``
and opened in the painter to inspect predictions.

Four supervision policies are supported so they can be compared deliberately:

``sparse`` (default)
    Copy the clinician's red/green corrective strokes exactly as exported.
    Untouched pixels stay unknown and contribute no loss.

``sparse_absent``
    As ``sparse``, except that every pixel of a scan the clinician decided
    *absent* is background. The decision already says the scan holds no
    lesion, so this needs no extra painting and no reviewed mask; it also
    brings in absent scans that have no strokes at all. Present scans keep
    their sparse strokes and uncertain scans stay unsupervised. Because it
    applies to validation as well, the validation loss then sees false
    positives anywhere on an absent scan.

``sparse_absent_val``
    ``sparse`` for training and ``sparse_absent`` for validation. Training is
    unchanged; only the signal that chooses which checkpoint to keep sees
    false positives on absent scans.

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
SUPERVISION_POLICIES = ('sparse', 'sparse_absent', 'sparse_absent_val',
                        'dense')
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


def patient_folds(label_rows, folds, seed=0):
    """Assign every patient to one of *folds* groups; returns ``{patient: fold}``.

    Patients with at least one scan decided present are dealt out separately
    from the rest, so positive patients are spread evenly over the folds. The
    assignment depends only on the patient list and the seed.
    """
    if folds < 2:
        raise ValueError('folds must be at least 2')
    has_positive = {}
    for row in label_rows:
        patient = row.get('patient_id') or row['filename']
        has_positive[patient] = (has_positive.get(patient, False)
                                 or row['decision'] == 'present')
    assignment = {}
    for positive in (True, False):
        patients = sorted(patient for patient, flag in has_positive.items()
                          if flag == positive)
        random.Random(f'{seed}|{positive}').shuffle(patients)
        for index, patient in enumerate(patients):
            assignment[patient] = index % folds
    return assignment


def fold_routes(label_rows, folds, fold, seed=0, train_per_val=5):
    """Route every scan to ``train``, ``val`` or ``test`` for one fold.

    Patients of fold *fold* are held out as ``test``: their scans get no
    annotation at all. The remaining patients are split again by patient, one
    part in ``train_per_val + 1`` becoming the validation set the trainer uses
    to choose checkpoints. No patient is in more than one role.
    """
    if not 0 <= fold < folds:
        raise ValueError(f'fold must be between 0 and {folds - 1}')

    def patient_of(row):
        return row.get('patient_id') or row['filename']

    outer = patient_folds(label_rows, folds, seed)
    remaining = [row for row in label_rows if outer[patient_of(row)] != fold]
    inner = patient_folds(remaining, train_per_val + 1,
                          seed=f'{seed}|inner|{fold}')
    route = {}
    for row in label_rows:
        patient = patient_of(row)
        if outer[patient] == fold:
            route[row['filename']] = 'test'
        else:
            route[row['filename']] = 'val' if inner[patient] == 0 else 'train'
    return route


def painter_project_order():
    """The painter's own ordering and split rules, loaded from its source.

    Loaded by path so a replayed session uses exactly the rule the painter
    applies, without putting the painter's modules on ``sys.path``.
    """
    import importlib.util
    path = (Path(__file__).resolve().parents[2] / 'painter' / 'src' / 'main'
            / 'python' / 'project_order.py')
    spec = importlib.util.spec_from_file_location('painter_project_order',
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def session_routes(label_rows, folds, fold, seed=0, train_per_val=5):
    """Route scans as a live painter session on one fold would.

    The held-out patients are those of ``fold_routes``, so a replayed session
    is scored on the same scans as ordinary cross-validation. The remaining
    scans are taken in the painter's seeded navigation order and split by the
    painter's patient-aware rule, which assigns each patient as it first
    appears: validation scans therefore arrive throughout the session, as
    they do for a user.
    """
    if not 0 <= fold < folds:
        raise ValueError(f'fold must be between 0 and {folds - 1}')
    patient_of = {row['filename']: row.get('patient_id') or row['filename']
                  for row in label_rows}
    outer = patient_folds(label_rows, folds, seed)
    ordered = seeded_file_order(list(patient_of), seed)
    session = [name for name in ordered if outer[patient_of[name]] != fold]
    train, val = painter_project_order().patient_grouped_train_val_split(
        session, patient_of, train_per_val)
    route = dict.fromkeys(ordered, 'test')
    route.update(dict.fromkeys(train, 'train'))
    route.update(dict.fromkeys(val, 'val'))
    return route


def reference_is_reliable(mask, max_foreground_fraction, max_components):
    """Return whether a positive reviewed mask is plausible as a dense label."""
    if not mask.any():
        return False
    if mask.mean() > max_foreground_fraction:
        return False
    from skimage.measure import label
    return int(label(mask, connectivity=2).max()) <= max_components


def populate_sparse(dataset, project, label_rows, route=None):
    """Copy the clinician's strokes into the project.

    *route* maps a file name to ``train`` or ``val``; any other value leaves
    the scan out (a held-out cross-validation fold). The default is the
    dataset's own split.
    """
    counts = Counter()
    stored = {Path(row['filename']).stem: row for row in label_rows}
    for split in ('train', 'val'):
        for source in sorted((dataset / 'annotations' / split).glob('*.png')):
            row = stored.get(source.stem)
            if row is None or row['split'] != split:
                raise ValueError(
                    'An annotation is stored under a split that disagrees '
                    'with bscan_labels.csv')
            target = route[row['filename']] if route else split
            if target not in ('train', 'val'):
                counts['held_out_annotation_files'] += 1
                continue
            foreground, background = read_annotation(source)
            shutil.copy2(source,
                         project / 'annotations' / target / source.name)
            counts[f'{target}_files'] += 1
            counts[f'{target}_foreground_files'] += int(foreground.any())
            counts[f'{target}_background_files'] += int(background.any())
    return counts


def populate_sparse_absent(dataset, project, label_rows,
                           splits=('train', 'val'), route=None):
    """Sparse strokes, plus whole-scan background for scans decided absent.

    *splits* limits the whole-scan labels to those splits; the others keep
    plain sparse strokes.
    """
    import numpy as np
    from PIL import Image

    counts = populate_sparse(dataset, project, label_rows, route)
    for row in label_rows:
        split = route[row['filename']] if route else row['split']
        if row['decision'] != 'absent' or split not in splits:
            continue
        destination = (project / 'annotations' / split
                       / (Path(row['filename']).stem + '.png'))
        had_strokes = destination.is_file()
        if had_strokes and read_annotation(destination)[0].any():
            raise ValueError(
                'A scan decided absent has foreground strokes')
        with Image.open(dataset / 'images' / row['filename']) as image:
            width, height = image.size
        background = np.ones((height, width), dtype=bool)
        write_annotation(destination, ~background, background)
        counts[f'{split}_absent_whole_scan_files'] += 1
        if not had_strokes:
            counts[f'{split}_files'] += 1
            counts[f'{split}_background_files'] += 1
            counts[f'{split}_absent_scans_added'] += 1
    return counts


def populate_dense(dataset, project, max_foreground_fraction, max_components,
                   route=None):
    counts = Counter()
    ground_truth = dataset / 'ground_truth'
    for row in read_rows(ground_truth / 'reviewed_mask_index.csv'):
        split = (route.get(row['filename'], row['split']) if route
                 else row['split'])
        if split not in ('train', 'val'):
            counts['held_out_reviewed_scans'] += 1
            continue
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
        sparse = dataset / 'annotations' / row['split'] / (stem + '.png')
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
                            dense_max_components=10, folds=None, fold=None,
                            session_split=False):
    """Build the project. With *folds* and *fold*, build one cross-validation
    fold: that fold's patients are held out and the rest are split by patient
    into training and validation, replacing the dataset's own split. With
    *session_split* as well, that split is the one the painter would make
    during a live session (see ``session_routes``)."""
    dataset = Path(dataset).resolve()
    sync_dir = Path(sync_dir).resolve()
    if (folds is None) != (fold is None):
        raise ValueError('Give both folds and fold, or neither')
    if session_split and folds is None:
        raise ValueError('A session split needs folds and fold')
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Unknown model type: {model_type}')
    if supervision not in SUPERVISION_POLICIES:
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

    if folds is None:
        split_of = {row['filename']: row['split'] for row in label_rows}
        split_method = 'dataset_patient_split_v1'
    elif session_split:
        split_of = session_routes(label_rows, folds, fold, seed)
        split_method = 'patient_fold_session_split_v1'
    else:
        split_of = fold_routes(label_rows, folds, fold, seed)
        split_method = 'patient_fold_v1'

    project = sync_dir / 'projects' / name
    if project.exists():
        raise ValueError(
            f'Project already exists; refusing to overwrite: {project}')
    for folder in PROJECT_SUBFOLDERS:
        (project / folder).mkdir(parents=True)

    if supervision == 'sparse':
        counts = populate_sparse(dataset, project, label_rows, split_of)
    elif supervision == 'sparse_absent':
        counts = populate_sparse_absent(dataset, project, label_rows,
                                        route=split_of)
    elif supervision == 'sparse_absent_val':
        counts = populate_sparse_absent(dataset, project, label_rows,
                                        splits=('val',), route=split_of)
    else:
        counts = populate_dense(dataset, project,
                                dense_max_foreground_fraction,
                                dense_max_components, route=split_of)

    ordered = seeded_file_order(
        [row['filename'] for row in label_rows], seed)
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
        'split_method': split_method,
        'model_type': model_type,
        'review_schema_version': 1,
        'auto_start_training_after_reviews': 10,
    }
    if folds is not None:
        # Held out from this fold: never annotated, scored after training.
        project_info['test_file_names'] = [
            f for f in ordered if split_of[f] == 'test']
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
    if folds is not None:
        roles = Counter(split_of.values())
        summary.update(folds=folds, fold=fold, session_split=session_split,
                       split_images={
                           role: roles[role]
                           for role in ('train', 'val', 'test')})
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
                        choices=SUPERVISION_POLICIES)
    parser.add_argument(
        '--dense-max-foreground-fraction', type=float, default=0.01,
        help='dense only: a positive reviewed mask covering more of the scan '
             'than this is treated as unreliable (default 0.01)')
    parser.add_argument(
        '--dense-max-components', type=int, default=10,
        help='dense only: a positive reviewed mask with more connected '
             'components than this is treated as unreliable (default 10)')
    parser.add_argument('--folds', type=int, default=None,
                        help='cross-validation: number of patient-level folds')
    parser.add_argument('--fold', type=int, default=None,
                        help='cross-validation: the fold to hold out '
                             '(0 .. folds-1)')
    parser.add_argument('--session-split', action='store_true',
                        help='with --folds: split the scans that are not held '
                             'out the way the painter does during a live '
                             'session (for run_session_replay.py)')
    args = parser.parse_args()
    summary = prepare_offline_project(
        args.dataset, args.sync_dir, args.name, model_type=args.model_type,
        seed=args.seed, supervision=args.supervision,
        dense_max_foreground_fraction=args.dense_max_foreground_fraction,
        dense_max_components=args.dense_max_components,
        folds=args.folds, fold=args.fold, session_split=args.session_split)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
