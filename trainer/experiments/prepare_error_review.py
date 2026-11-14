"""Collect the scans several models get wrong into a small painter project.

When different models, losses and training lengths all miss the same positive
scans and flag the same negative ones, the cause is more likely in the scans
or their labels than in the training. This tool finds those scans from
existing evaluations and builds a review project a person can step through in
the painter:

* ``missed_positive``  scans decided present that at most a third of the
  models flag;
* ``found_positive``   a few scans decided present that every model flags, as
  a reference for what the models do recognise;
* ``flagged_negative`` scans decided absent that at least half the models
  flag.

"Flag" is decided per model at a matched operating point: each model's own
threshold on its highest lesion probability that clears ``--target-specificity``
of the absent scans (default 0.9). Fixed thresholds would mostly compare how
conservative the models are.

The project holds only those scans, in that order. Each shows the clinician's
own strokes and one model's overlay. It is for looking, not training: it has
no training annotations and never starts training.

Scan file names embed patient IDs, so the list of scans is written to
``<project>/review_list.csv`` and never printed. CLI output is counts only.

    python prepare_error_review.py --name review_errors_val \\
        --sync-dir D:\\RootPainterSync --overlay-project <project> \\
        --evaluations <evaluation dir> <evaluation dir> ...
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from mask_io import write_segmentation
from prepare_offline_project import PROJECT_SUBFOLDERS, read_rows

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
GROUPS = ('missed_positive', 'found_positive', 'flagged_negative')


def calls_at_specificity(rows, target_specificity=0.9):
    """Flag scans at the threshold that clears *target_specificity* negatives.

    *rows* are ``per_image.csv`` records. Returns ``{file name: flagged}``.
    """
    negative_scores = sorted(float(row['max_prob']) for row in rows
                             if row['decision'] == 'absent')
    if not negative_scores:
        raise ValueError('An evaluation has no scan decided absent')
    index = math.ceil(target_specificity * len(negative_scores)) - 1
    cut = negative_scores[min(max(index, 0), len(negative_scores) - 1)]
    return {row['filename']: float(row['max_prob']) > cut for row in rows}


def count_flags(evaluations, target_specificity=0.9):
    """Count, for every scan, how many evaluations flag it.

    *evaluations* is a list of ``per_image.csv`` row lists. They must cover
    the same scans. Returns ``(decision by file name, flag count by file
    name)``.
    """
    decision = {row['filename']: row['decision'] for row in evaluations[0]}
    flags = Counter({name: 0 for name in decision})
    for rows in evaluations:
        if {row['filename'] for row in rows} != set(decision):
            raise ValueError('The evaluations do not cover the same scans')
        for name, flagged in calls_at_specificity(
                rows, target_specificity).items():
            flags[name] += int(flagged)
    return decision, flags


def select_review_scans(decision, flags, models, max_found=5):
    """Pick and order the scans worth a person's look."""
    missed = sorted((name for name, label in decision.items()
                     if label == 'present' and flags[name] <= models // 3),
                    key=lambda name: (flags[name], name))
    found = sorted(name for name, label in decision.items()
                   if label == 'present' and flags[name] == models)
    flagged = sorted((name for name, label in decision.items()
                      if label == 'absent' and flags[name] >= models / 2),
                     key=lambda name: (-flags[name], name))
    scans = []
    for group, names in (('missed_positive', missed),
                         ('found_positive', found[:max_found]),
                         ('flagged_negative', flagged)):
        scans.extend(dict(filename=name, clinician_decision=decision[name],
                          group=group, models_flagging=flags[name],
                          models_compared=models) for name in names)
    return scans


def create_review_project(dataset, sync_dir, name, scans, predict,
                          provenance=None):
    """Write the review project. ``predict(file name)`` returns a mask.

    The painter asks the trainer to segment the current scan on every step. A
    running trainer leaves existing overlays alone, but creates a first model
    in a project that has none, so the project is declared as ``unet``: the
    cheapest model to create, and one that never loads foundation weights.
    Which model drew the overlays is recorded in ``error_review.json``.
    """
    dataset = Path(dataset).resolve()
    sync_dir = Path(sync_dir).resolve()
    if not scans:
        raise ValueError('No scan was selected for review')
    project = sync_dir / 'projects' / name
    if project.exists():
        raise ValueError(
            f'Project already exists; refusing to overwrite: {project}')
    split_of = {row['filename']: row['split'] for row in
                read_rows(dataset / 'ground_truth' / 'bscan_labels.csv')}
    for folder in PROJECT_SUBFOLDERS:
        (project / folder).mkdir(parents=True)

    with_strokes = 0
    for scan in scans:
        stem = Path(scan['filename']).stem
        strokes = (dataset / 'annotations' / split_of[scan['filename']]
                   / (stem + '.png'))
        if strokes.is_file():
            # Shown, not trained on: the project has no training split.
            shutil.copy2(strokes, project / 'annotations' / 'val' / strokes.name)
            with_strokes += 1
        write_segmentation(project / 'segmentations' / (stem + '.png'),
                           predict(scan['filename']))

    ordered = [scan['filename'] for scan in scans]
    images_dir = dataset / 'images'
    project_info = {
        'name': name,
        'dataset': Path(
            os.path.relpath(images_dir, sync_dir / 'datasets')).as_posix(),
        'original_model_file': 'random weights',
        'location': str(PurePosixPath('projects') / name),
        'file_names': ordered,
        'image_order_method': 'error_review_v1',
        'dataset_manifest': None,
        'image_order_seed': 0,
        'training_seed': 0,
        'train_file_names': [],
        'val_file_names': ordered,
        'split_method': 'error_review_v1',
        'model_type': 'unet',
        'review_schema_version': 1,
        # Review only: zero disables the painter's automatic training start.
        'auto_start_training_after_reviews': 0,
        # Open on the first scan, not the last one that has strokes.
        'image_index': 0,
    }
    with open(project / (name + '.seg_proj'), 'w', encoding='utf-8') as stream:
        json.dump(project_info, stream, indent=4)

    fields = ['position', 'filename', 'clinician_decision', 'group',
              'models_flagging', 'models_compared']
    with open(project / 'review_list.csv', 'w', encoding='utf-8',
              newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, scan in enumerate(scans, start=1):
            writer.writerow(dict(scan, position=position))

    counts = Counter(scan['group'] for scan in scans)
    positions = {}
    start = 1
    for group in GROUPS:
        if counts[group]:
            positions[group] = [start, start + counts[group] - 1]
            start += counts[group]
    summary = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(project),
        'dataset': str(dataset),
        'scans': len(scans),
        'group_counts': {group: counts[group] for group in GROUPS},
        'group_positions': positions,
        'scans_with_clinician_strokes': with_strokes,
        'provenance': provenance or {},
    }
    with open(project / 'error_review.json', 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    return summary


def make_predictor(overlay_project, checkpoint, dataset, batch_size=12,
                   threshold=0.5):
    """Return ``(predict, model type, checkpoint name)`` for one checkpoint."""
    from evaluate_project import model_geometry, resolve_checkpoint

    overlay_project = Path(overlay_project).resolve()
    project_files = sorted(overlay_project.glob('*.seg_proj'))
    if len(project_files) != 1:
        raise SystemExit(f'Expected one .seg_proj file in {overlay_project}')
    with open(project_files[0], encoding='utf-8') as stream:
        model_type = json.load(stream).get('model_type') or 'unet'
    checkpoint = resolve_checkpoint(overlay_project, checkpoint)

    sys.path.insert(0, str(SRC_DIR))
    import im_utils  # pylint: disable=import-outside-toplevel
    import model_utils  # pylint: disable=import-outside-toplevel

    in_w, out_w = model_geometry(model_type)
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        with contextlib.redirect_stdout(devnull):
            model = model_utils.load_model(str(checkpoint),
                                           model_type=model_type)

    def predict(filename):
        image = im_utils.load_image(str(Path(dataset) / 'images' / filename))
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                probs = model_utils.ensemble_foreground_probs(
                    [model], image, batch_size, in_w, out_w)
        return probs > threshold

    return predict, model_type, checkpoint.name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--evaluations', required=True, nargs='+', type=Path,
                        help='evaluation directories (each with per_image.csv '
                             'for the same scans)')
    parser.add_argument('--overlay-project', required=True, type=Path,
                        help='project whose model draws the overlays')
    parser.add_argument('--overlay-checkpoint', default=None,
                        help='checkpoint name or path (default: the '
                             "overlay project's validation-best)")
    parser.add_argument('--name', required=True, help='new review project')
    parser.add_argument('--sync-dir', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, default=None,
                        help='curated dataset (default: the one recorded in '
                             'the first evaluation)')
    parser.add_argument('--target-specificity', type=float, default=0.9)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='probability threshold of the overlay')
    parser.add_argument('--max-found', type=int, default=5,
                        help='reference positives every model flags')
    parser.add_argument('--gpu', default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    evaluation_dirs = [path.resolve() for path in args.evaluations]
    evaluations = [read_rows(path / 'per_image.csv')
                   for path in evaluation_dirs]
    dataset = args.dataset
    if dataset is None:
        with open(evaluation_dirs[0] / 'metrics.json',
                  encoding='utf-8') as stream:
            dataset = Path(json.load(stream)['dataset'])
    if (args.sync_dir.resolve() / 'projects' / args.name).exists():
        raise SystemExit(f'Project already exists: {args.name}')

    decision, flags = count_flags(evaluations, args.target_specificity)
    scans = select_review_scans(decision, flags, len(evaluations),
                                args.max_found)
    predict, model_type, checkpoint = make_predictor(
        args.overlay_project, args.overlay_checkpoint, dataset,
        threshold=args.threshold)
    summary = create_review_project(
        dataset, args.sync_dir, args.name, scans, predict,
        provenance={
            'overlay_model_type': model_type,
            'evaluations': [str(path) for path in evaluation_dirs],
            'target_specificity': args.target_specificity,
            'overlay_project': str(args.overlay_project.resolve()),
            'overlay_checkpoint': checkpoint,
            'overlay_threshold': args.threshold,
        })
    print(json.dumps({key: summary[key] for key in (
        'project', 'scans', 'group_counts', 'group_positions',
        'scans_with_clinician_strokes')}, indent=2))


if __name__ == '__main__':
    main()
