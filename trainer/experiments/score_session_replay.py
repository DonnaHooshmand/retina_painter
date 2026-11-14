"""Score a replayed annotation session: detection against scans viewed.

``run_session_replay.py`` records, at each milestone of a session, the model
the painter would be showing. Here each of those models is scored on the
patients its fold held out, and the folds are pooled milestone by milestone.
The result is the curve RetinaPainter is about: how good the model on screen
is after 25, 50, 100, ... scans and how many minutes that took.

Scoring is blob-level, as in ``score_cross_validation.py`` (a RIPL is there or
not), with one difference. A session has almost no validation scans early on,
so no cut-off can be calibrated on them. Instead:

* **Ranking** (AUROC of the blob score) needs no cut-off.
* **Flagged / cleared / marks found / false blobs** use the painter's own
  cut-off (``--threshold``, default 0.5): this is what the user would see.
* **Flagged with 90% cleared** reads the same model at the cut-off that
  leaves ``--target-specificity`` of the held-out scans decided absent
  unflagged. The cut-off is chosen on the scans it is reported on, so it
  describes the model's ranking at a fixed strictness and is not an honest
  operating point. It is blank when no cut-off clears that many.

Intervals are patient-level bootstrap intervals. ``--baseline`` adds paired
differences against another replay scored on the same scans. Per-scan rows go
to ``<project>/evaluation/replay__<model>/held_out_scores.csv`` and are never
printed (file names embed patient IDs).

    python score_session_replay.py --sync-dir D:\\RootPainterSync \\
        --name replay_retfound_rfa --gpu 0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_project import json_safe, read_rows
from score_cross_validation import (FIELDS, GRID, FoldScorer, bootstrap,
                                    compare_runs, fold_projects, summarize)

COMPARED = ('auroc', 'sensitivity', 'specificity', 'lesion_recall',
            'fp_blobs_per_negative_scan')


def read_milestones(project_dir):
    path = Path(project_dir) / 'replay' / 'milestones.jsonl'
    if not path.is_file():
        raise SystemExit(f'{Path(project_dir).name} holds no replayed session')
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sensitivity_at_specificity(records, target=0.9, grid=GRID):
    """Positive scans flagged at the cut-off that clears *target* negatives.

    Returns ``(flagged, positives)``; ``flagged`` is ``None`` when no cut-off
    on the grid leaves that share of the scans decided absent unflagged, as
    for a model that puts a confident blob on nearly every scan.
    """
    positive = np.array([float(r['blob_score']) for r in records
                         if r['decision'] == 'present'])
    negative = np.array([float(r['blob_score']) for r in records
                         if r['decision'] == 'absent'])
    if len(negative):
        for threshold in grid:
            if np.mean(negative < threshold) >= target:
                return int((positive >= threshold).sum()), len(positive)
    return None, len(positive)


def score_model(project_dir, model, threshold=0.5, min_px=25, batch_size=12,
                rescore=False):
    """Held-out records for one recorded model, stored beside the project."""
    project_dir = Path(project_dir).resolve()
    model_path = project_dir / model
    out_dir = project_dir / 'evaluation' / f'replay__{model_path.stem}'
    settings = dict(model=model, threshold=threshold, min_lesion_px=min_px)
    if not rescore and (out_dir / 'scoring.json').is_file():
        with open(out_dir / 'scoring.json', encoding='utf-8') as stream:
            if json.load(stream) == settings:
                return read_rows(out_dir / 'held_out_scores.csv')
    scorer = FoldScorer(project_dir, model_path, min_px, batch_size)
    records = [scorer.held_out(name, threshold) for name in
               scorer.definite(scorer.settings['test_file_names'])]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'held_out_scores.csv', 'w', encoding='utf-8',
              newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    with open(out_dir / 'scoring.json', 'w', encoding='utf-8') as stream:
        json.dump(settings, stream, indent=2)
    return records


def score_replay_fold(project_dir, **options):
    """One fold's milestones, each with its model's held-out records."""
    milestones = read_milestones(project_dir)
    for milestone in milestones:
        milestone['records'] = score_model(project_dir, milestone['model'],
                                           **options)
    return milestones


def pool(folds, target_specificity=0.9, resamples=2000):
    """Pool scored folds milestone by milestone; returns one row each."""
    labels = [milestone['milestone'] for milestone in folds[0]]
    if any([milestone['milestone'] for milestone in fold] != labels
           for fold in folds):
        raise ValueError('The folds did not record the same milestones')
    rows = []
    for index, label in enumerate(labels):
        parts = [fold[index] for fold in folds]
        records = [record for part in parts for record in part['records']]
        intervals, patients = bootstrap(records, resamples=resamples)
        at_target = [sensitivity_at_specificity(part['records'],
                                                target_specificity)[0]
                     for part in parts]

        def mean(key):
            return float(np.mean([part[key] for part in parts]))

        rows.append(dict(
            summarize(records), milestone=label, folds=len(parts),
            patients=patients, intervals=intervals,
            scans_viewed=mean('scans_viewed'),
            session_minutes=mean('session_minutes'),
            annotated_scans=float(np.mean(
                [part['train_annotations'] + part['val_annotations']
                 for part in parts])),
            annotated_scans_with_lesion_marks=float(np.mean(
                [part['train_with_lesion_marks']
                 + part['val_with_lesion_marks'] for part in parts])),
            showing=dict(Counter(part['ui_model'] for part in parts)),
            positive_scans_flagged_at_target_specificity=(
                None if None in at_target else int(sum(at_target))),
            records=records))
    return rows


def compare_replays(rows, baseline_rows):
    """Paired differences (run minus baseline) at every shared milestone."""
    theirs = {row['milestone']: row for row in baseline_rows}
    compared = []
    for row in rows:
        other = theirs.get(row['milestone'])
        if other is None:
            continue
        versus = compare_runs(row['records'], other['records'])
        compared.append(dict(milestone=row['milestone'],
                             **{key: versus[key] for key in COMPARED}))
    return compared


def score_run(sync_dir, name, target_specificity=0.9, **options):
    folds = [score_replay_fold(project, **options)
             for project in fold_projects(sync_dir, name)]
    return pool(folds, target_specificity)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sync-dir', required=True, type=Path)
    parser.add_argument('--name', required=True,
                        help='run name; folds are <name>_fold0, _fold1, ...')
    parser.add_argument('--project', type=Path, default=None,
                        help='score this one fold project only and stop '
                             '(fills the stored scores; nothing is pooled)')
    parser.add_argument('--baseline', default=None,
                        help='another replay run to compare against')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="cut-off for what the user sees (the painter's "
                             'is 0.5)')
    parser.add_argument('--target-specificity', type=float, default=0.9)
    parser.add_argument('--min-lesion-px', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--rescore', action='store_true',
                        help='recompute stored scores')
    parser.add_argument('--gpu', default=None)
    return parser.parse_args(argv)


def _share(count, total):
    return f'{count}/{total} {count / total:4.0%}' if total else 'n/a'


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    options = dict(threshold=args.threshold, min_px=args.min_lesion_px,
                   batch_size=args.batch_size, rescore=args.rescore)
    if args.project is not None:
        scored = score_replay_fold(args.project, **options)
        print(f'{args.project.name}: {len(scored)} milestones scored')
        return

    rows = score_run(args.sync_dir, args.name, args.target_specificity,
                     **options)
    summary = dict(run=args.name, threshold=args.threshold,
                   target_specificity=args.target_specificity,
                   min_lesion_px=args.min_lesion_px)
    if args.baseline:
        options['rescore'] = False
        baseline = score_run(args.sync_dir, args.baseline,
                             args.target_specificity, **options)
        summary['versus'] = dict(baseline=args.baseline,
                                 milestones=compare_replays(rows, baseline))
    summary['milestones'] = [
        {key: value for key, value in row.items() if key != 'records'}
        for row in rows]
    out_path = (args.sync_dir.resolve() / 'projects'
                / f'{args.name}_session_replay.json')
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)

    first = rows[0]
    print(f"{args.name}: {first['folds']} folds, {first['scans']} held-out "
          f"scans from {first['patients']} patients "
          f"({first['positive_scans']} positive, "
          f"{first['negative_scans']} negative); model on screen scored at "
          f'cut-off {args.threshold}')
    print('  scans  minutes  annotated  with marks  ranking [95% range]     '
          'flagged       cleared       flagged, 90% cleared  marks found   '
          'false blobs/neg scan')
    for row in rows:
        low, high = row['intervals']['auroc']
        at_target = row['positive_scans_flagged_at_target_specificity']
        print(f"  {row['scans_viewed']:5.0f}  {row['session_minutes']:7.1f}  "
              f"{row['annotated_scans']:9.1f}  "
              f"{row['annotated_scans_with_lesion_marks']:10.1f}  "
              f"{row['auroc']:.3f} [{low:.3f}, {high:.3f}]  "
              f"{_share(row['positive_scans_flagged'], row['positive_scans']):12s}  "
              f"{_share(row['negative_scans_cleared'], row['negative_scans']):12s}  "
              f"{'n/a' if at_target is None else _share(at_target, row['positive_scans']):20s}  "
              f"{_share(row['lesions_found'], row['lesions']):12s}  "
              f"{row['fp_blobs_per_negative_scan']:.2f}")
    if args.baseline:
        print(f'  versus {args.baseline} (this run minus baseline), ranking '
              'and scans flagged / cleared at the cut-off:')
        for row in summary['versus']['milestones']:
            parts = []
            for key in ('auroc', 'sensitivity', 'specificity'):
                low, high = row[key]['interval']
                parts.append(f"{row[key]['difference']:+.3f} "
                             f'[{low:+.3f}, {high:+.3f}]')
            print(f"  {str(row['milestone']):>5s}  " + '   '.join(parts))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
