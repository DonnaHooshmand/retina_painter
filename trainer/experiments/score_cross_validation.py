"""Score a patient-level cross-validation run on blobs, not pixels.

A RIPL is either there or not, and the clinician marks one with a dot, not an
outline. So nothing here measures pixel overlap. Each fold project (made with
``prepare_offline_project.py --folds K --fold k`` and trained as usual) is
scored on the patients it never saw, and the folds are pooled so that every
scan with a definite decision counts once.

Per scan the model gets one number, the **blob score**: the highest
probability threshold at which its prediction still contains a connected blob
of at least ``--min-lesion-px`` pixels (0 when it never does). Then:

* **AUROC** of the blob score: how well scans decided present are ranked
  above scans decided absent, with no threshold chosen.
* **Operating point**: each fold's threshold is set on that fold's own
  validation scans, as the lowest one that leaves ``--target-specificity``
  (default 0.9) of the absent ones unflagged, and is then applied unchanged to
  the held-out scans. Sensitivity and specificity on the held-out scans are
  therefore honest: the threshold never saw them.
* **Lesions found**: clinician marks touched by a predicted blob at that
  threshold. **False-positive blobs**: predicted blobs on absent scans.

Intervals are patient-level bootstrap intervals. ``--baseline`` adds paired
differences against another run scored on the same scans.

Per-scan rows go to ``<project>/evaluation/fold__<checkpoint>/fold_scores.csv``
and are never printed (file names embed patient IDs).

    python score_cross_validation.py --sync-dir D:\\RootPainterSync \\
        --name cv_sparse_retfound_rfa --gpu 0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_project import (auroc, image_record, json_safe, label_components,
                              model_geometry, read_rows, resolve_checkpoint)
from mask_io import read_annotation

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
FIELDS = ['filename', 'patient_id', 'role', 'decision', 'blob_score',
          'max_prob', 'flagged', 'lesions', 'lesions_found', 'fp_blobs']


# --------------------------------------------------------------------------
# blob-level scoring (pure numpy; unit-tested without a model)
# --------------------------------------------------------------------------

def blob_score(probs, min_px=25, grid=GRID):
    """Highest threshold at which *probs* still holds a blob of *min_px*.

    Returns 0.0 when no threshold on the grid leaves such a blob.
    """
    probs = np.asarray(probs)
    top = float(probs.max())
    for threshold in grid[::-1]:
        if threshold >= top:
            continue
        mask = probs > threshold
        if mask.sum() < min_px:
            continue
        areas = label_components(mask)[1]
        if len(areas) and areas.max() >= min_px:
            return float(threshold)
    return 0.0


def calibrate_threshold(negative_scores, target_specificity=0.9, grid=GRID):
    """Lowest threshold leaving *target_specificity* of negatives unflagged.

    A scan is flagged when its blob score is at least the threshold.
    """
    scores = np.asarray(list(negative_scores), dtype=float)
    if not len(scores):
        raise ValueError('No scan decided absent to calibrate on')
    for threshold in grid:
        if np.mean(scores < threshold) >= target_specificity:
            return float(threshold)
    return float(grid[-1])


def summarize(records):
    """Pooled blob-level metrics for held-out scan records."""
    positive = [r for r in records if r['decision'] == 'present']
    negative = [r for r in records if r['decision'] == 'absent']

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float('nan')

    found = sum(int(r['flagged']) for r in positive)
    cleared = sum(not int(r['flagged']) for r in negative)
    lesions = sum(int(r['lesions']) for r in positive)
    sensitivity = ratio(found, len(positive))
    specificity = ratio(cleared, len(negative))
    return {
        'scans': len(records),
        'positive_scans': len(positive),
        'negative_scans': len(negative),
        'positive_scans_flagged': found,
        'negative_scans_cleared': cleared,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'auroc': auroc([float(r['blob_score']) for r in positive + negative],
                       [1] * len(positive) + [0] * len(negative)),
        'lesions': lesions,
        'lesions_found': sum(int(r['lesions_found']) for r in positive),
        'lesion_recall': ratio(
            sum(int(r['lesions_found']) for r in positive), lesions),
        'fp_blobs_per_negative_scan': ratio(
            sum(int(r['fp_blobs']) for r in negative), len(negative)),
    }


HEADLINE = ('sensitivity', 'specificity', 'balanced_accuracy', 'auroc',
            'lesion_recall', 'fp_blobs_per_negative_scan')


def bootstrap(records, baseline=None, resamples=2000, seed=0):
    """Patient-level 95% intervals; with *baseline*, for the difference."""
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.get('patient_id') or record['filename']].append(index)
    groups = list(groups.values())
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(resamples):
        chosen = [index for group in rng.integers(0, len(groups), len(groups))
                  for index in groups[group]]
        values = summarize([records[i] for i in chosen])
        if baseline is not None:
            reference = summarize([baseline[i] for i in chosen])
            values = {key: values[key] - reference[key] for key in HEADLINE}
        for key in HEADLINE:
            samples[key].append(values[key])
    intervals = {}
    for key, values in samples.items():
        values = np.array(values, dtype=float)
        values = values[np.isfinite(values)]
        intervals[key] = ([float(np.percentile(values, 2.5)),
                           float(np.percentile(values, 97.5))]
                          if len(values) else [float('nan')] * 2)
    return intervals, len(groups)


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------

def load_fold(project_dir):
    project_dir = Path(project_dir).resolve()
    project_files = sorted(project_dir.glob('*.seg_proj'))
    if len(project_files) != 1:
        raise SystemExit(f'Expected one .seg_proj file in {project_dir}')
    with open(project_files[0], encoding='utf-8') as stream:
        settings = json.load(stream)
    if 'test_file_names' not in settings:
        raise SystemExit(f'{project_dir.name} is not a cross-validation fold '
                         '(it has no held-out scans)')
    with open(project_dir / 'offline_experiment.json',
              encoding='utf-8') as stream:
        experiment = json.load(stream)
    return project_dir, settings, experiment


class FoldScorer:
    """Scores scans of one fold project with one model file.

    Probabilities come from the painter's own inference path. Only scans with
    a definite decision are scored; the reference for a scan decided present
    is the clinician's own lesion marks.
    """

    def __init__(self, project_dir, model_path, min_px=25, batch_size=12):
        self.project_dir, self.settings, self.experiment = load_fold(
            project_dir)
        self.dataset = Path(self.experiment['dataset'])
        self.labels = {
            row['filename']: row for row in
            read_rows(self.dataset / 'ground_truth' / 'bscan_labels.csv')}
        self.min_px = min_px
        self.batch_size = batch_size
        model_type = self.settings.get('model_type') or 'unet'

        sys.path.insert(0, str(SRC_DIR))
        import im_utils  # pylint: disable=import-outside-toplevel
        import model_utils  # pylint: disable=import-outside-toplevel

        self.im_utils, self.model_utils = im_utils, model_utils
        self.in_w, self.out_w = model_geometry(model_type)
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                self.model = model_utils.load_model(str(model_path),
                                                    model_type=model_type)

    def probabilities(self, filename):
        image = self.im_utils.load_image(
            str(self.dataset / 'images' / filename))
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                return self.model_utils.ensemble_foreground_probs(
                    [self.model], image, self.batch_size, self.in_w,
                    self.out_w)

    def definite(self, names):
        return [name for name in names
                if self.labels[name]['decision'] in ('present', 'absent')]

    def ranked(self, name, role):
        """A scan's blob score only (no threshold applied yet)."""
        probs = self.probabilities(name)
        label = self.labels[name]
        return dict(
            filename=name, patient_id=label.get('patient_id', ''), role=role,
            decision=label['decision'],
            blob_score=blob_score(probs, self.min_px),
            max_prob=float(probs.max()),
            flagged='', lesions='', lesions_found='', fp_blobs='')

    def held_out(self, name, threshold):
        """A held-out scan called at *threshold*, with its blob counts."""
        probs = self.probabilities(name)
        label = self.labels[name]
        reference = np.zeros(probs.shape, dtype=bool)
        if label['decision'] == 'present':
            strokes = (self.dataset / 'annotations' / label['split']
                       / (Path(name).stem + '.png'))
            if strokes.is_file():
                reference = read_annotation(strokes)[0]
        counted = image_record(probs, reference, threshold, self.min_px)
        score = blob_score(probs, self.min_px)
        return dict(
            filename=name, patient_id=label.get('patient_id', ''),
            role='test', decision=label['decision'], blob_score=score,
            max_prob=float(probs.max()), flagged=int(score >= threshold),
            lesions=counted['ref_lesions'],
            lesions_found=counted['ref_lesions_detected'],
            fp_blobs=counted['fp_lesions'])


def score_fold(project_dir, target_specificity=0.9, min_px=25, batch_size=12,
               checkpoint=None):
    """Score one fold's held-out scans; returns ``(records, metrics)``."""
    checkpoint = resolve_checkpoint(Path(project_dir).resolve(), checkpoint)
    scorer = FoldScorer(project_dir, checkpoint, min_px, batch_size)
    project_dir, settings = scorer.project_dir, scorer.settings
    experiment = scorer.experiment
    out_dir = project_dir / 'evaluation' / f'fold__{checkpoint.stem}'

    records = [scorer.ranked(name, 'val')
               for name in scorer.definite(settings['val_file_names'])]
    threshold = calibrate_threshold(
        (r['blob_score'] for r in records if r['decision'] == 'absent'),
        target_specificity)
    held_out = [scorer.held_out(name, threshold)
                for name in scorer.definite(settings['test_file_names'])]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'fold_scores.csv', 'w', encoding='utf-8',
              newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records + held_out)
    metrics = dict(summarize(held_out), project=project_dir.name,
                   fold=experiment.get('fold'), checkpoint=checkpoint.name,
                   threshold=threshold,
                   target_specificity=target_specificity, min_lesion_px=min_px)
    with open(out_dir / 'fold_metrics.json', 'w', encoding='utf-8') as stream:
        json.dump(json_safe(metrics), stream, indent=2, allow_nan=False)
    return held_out, metrics


def load_fold_scores(project_dir, checkpoint=None):
    """Read an already scored fold; ``None`` if it has not been scored."""
    project_dir = Path(project_dir).resolve()
    checkpoint = resolve_checkpoint(project_dir, checkpoint)
    out_dir = project_dir / 'evaluation' / f'fold__{checkpoint.stem}'
    if not (out_dir / 'fold_metrics.json').is_file():
        return None
    held_out = [row for row in read_rows(out_dir / 'fold_scores.csv')
                if row['role'] == 'test']
    with open(out_dir / 'fold_metrics.json', encoding='utf-8') as stream:
        return held_out, json.load(stream)


# --------------------------------------------------------------------------
# a whole run
# --------------------------------------------------------------------------

def fold_projects(sync_dir, name):
    projects = sorted((Path(sync_dir) / 'projects').glob(f'{name}_fold*'),
                      key=lambda path: int(path.name.rsplit('fold', 1)[1]))
    if not projects:
        raise SystemExit(f'No fold projects named {name}_fold* in {sync_dir}')
    return projects


def score_run(sync_dir, name, rescore=False, **options):
    """Score every fold of a run (reusing stored scores) and pool them."""
    records, folds = [], []
    for project in fold_projects(sync_dir, name):
        scored = None if rescore else load_fold_scores(project)
        if scored is None:
            scored = score_fold(project, **options)
        records.extend(scored[0])
        folds.append(scored[1])
    summary = summarize(records)
    intervals, patients = bootstrap(records)
    summary.update(
        run=name, folds=len(folds), patients=patients,
        intervals=intervals,
        mean_fold_auroc=float(np.mean([fold['auroc'] for fold in folds])),
        fold_summaries=[{key: fold[key] for key in (
            'fold', 'checkpoint', 'threshold', 'positive_scans',
            'positive_scans_flagged', 'negative_scans',
            'negative_scans_cleared', 'auroc')} for fold in folds])
    return records, summary


def compare_runs(records, baseline):
    """Paired differences (run minus baseline) on the same held-out scans."""
    by_name = {record['filename']: record for record in baseline}
    if set(by_name) != {record['filename'] for record in records}:
        raise ValueError('The two runs were not scored on the same scans')
    paired = [by_name[record['filename']] for record in records]
    ours, theirs = summarize(records), summarize(paired)
    intervals, _ = bootstrap(records, baseline=paired)
    return {key: {'run': ours[key], 'baseline': theirs[key],
                  'difference': ours[key] - theirs[key],
                  'interval': intervals[key]} for key in HEADLINE}


LABELS = {
    'sensitivity': 'positive scans flagged (sensitivity)',
    'specificity': 'negative scans cleared (specificity)',
    'balanced_accuracy': 'balanced accuracy',
    'auroc': 'AUROC of the blob score',
    'lesion_recall': 'clinician marks found',
    'fp_blobs_per_negative_scan': 'false-positive blobs per negative scan',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sync-dir', required=True, type=Path)
    parser.add_argument('--name', required=True,
                        help='run name; folds are <name>_fold0, _fold1, ...')
    parser.add_argument('--baseline', default=None,
                        help='another run name to compare against')
    parser.add_argument('--target-specificity', type=float, default=0.9)
    parser.add_argument('--min-lesion-px', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--rescore', action='store_true',
                        help='recompute fold scores even if stored')
    parser.add_argument('--gpu', default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    options = dict(target_specificity=args.target_specificity,
                   min_px=args.min_lesion_px, batch_size=args.batch_size)
    records, summary = score_run(args.sync_dir, args.name, args.rescore,
                                 **options)
    if args.baseline:
        baseline, _ = score_run(args.sync_dir, args.baseline, args.rescore,
                                **options)
        summary['versus'] = dict(baseline=args.baseline,
                                 **compare_runs(records, baseline))
    out_path = (args.sync_dir.resolve() / 'projects'
                / f'{args.name}_cross_validation.json')
    with open(out_path, 'w', encoding='utf-8') as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)

    print(f"{args.name}: {summary['folds']} folds, {summary['scans']} held-out "
          f"scans from {summary['patients']} patients "
          f"({summary['positive_scans']} positive, "
          f"{summary['negative_scans']} negative)")
    print(f"  positive scans flagged {summary['positive_scans_flagged']} of "
          f"{summary['positive_scans']}; negative scans cleared "
          f"{summary['negative_scans_cleared']} of {summary['negative_scans']}; "
          f"clinician marks found {summary['lesions_found']} of "
          f"{summary['lesions']}")
    for key in HEADLINE:
        low, high = summary['intervals'][key]
        print(f'  {LABELS[key]:42s} {summary[key]:6.3f}  '
              f'[{low:.3f}, {high:.3f}]')
    print(f"  mean per-fold AUROC {summary['mean_fold_auroc']:.3f}")
    for fold in summary['fold_summaries']:
        print(f"  fold {fold['fold']}: threshold {fold['threshold']:.2f}, "
              f"flagged {fold['positive_scans_flagged']}/{fold['positive_scans']}, "
              f"cleared {fold['negative_scans_cleared']}/{fold['negative_scans']}, "
              f"AUROC {fold['auroc']:.3f}")
    if args.baseline:
        print(f'  versus {args.baseline} (this run minus baseline):')
        for key in HEADLINE:
            row = summary['versus'][key]
            low, high = row['interval']
            print(f"  {LABELS[key]:42s} {row['baseline']:6.3f} -> "
                  f"{row['run']:6.3f}  {row['difference']:+.3f}  "
                  f'[{low:+.3f}, {high:+.3f}]')
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
