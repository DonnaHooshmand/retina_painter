"""Score a project's checkpoint against a curated dataset's reference labels.

Inference uses the painter's own path (``ensemble_foreground_probs``: tiled,
flip-averaged), so the numbers describe what a clinician would see as the
overlay. Three views are reported, from most to least trustworthy reference:

**B-scan level** — does the model flag the scan? A scan is called positive
when its largest predicted connected component has at least
``--min-lesion-px`` pixels. The reference is the clinician's present/absent
decision. Sensitivity and specificity carry Wilson 95% intervals; AUROC is
threshold-free (scored by largest predicted component and by peak probability).

**Lesion level** — are reference lesions hit, and how many false-positive
blobs appear per scan? Components smaller than ``--min-lesion-px`` are ignored
on both sides.

**Pixel level** — pooled precision / recall / F1 against
``ground_truth/reviewed_masks``. Those masks are the clinician's accepted
*corrected predictions*: they are exact for absent scans (all background) but
can retain uncorrected output of an earlier model on present scans, so pixel
F1 on positives is an agreement measure, not a certified contour accuracy.

Uncertain scans are excluded from every metric. Evaluating the split that
selected the checkpoint (``val``) is optimistically biased; it is a development
comparison, not a held-out test.

CLI output is aggregate only; per-image results stay in the output directory.

    python evaluate_project.py --project D:\\RootPainterSync\\projects\\NAME --gpu 0
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from mask_io import read_binary_mask, write_segmentation

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
BSCAN_SWEEP_PX = (1, 10, 25, 50, 100, 200)


# --------------------------------------------------------------------------
# metrics (pure numpy; unit-tested without a model)
# --------------------------------------------------------------------------

def label_components(mask):
    """Return ``(labels, areas)`` for the 8-connected components of *mask*."""
    from skimage.measure import label
    labels = label(np.asarray(mask, dtype=bool), connectivity=2)
    return labels, np.bincount(labels.ravel())[1:]


def image_record(probs, reference, threshold=0.5, min_lesion_px=25):
    """Per-image counts for one probability map and its reference mask."""
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(probs) > threshold
    pred_labels, pred_areas = label_components(predicted)
    ref_labels, ref_areas = label_components(reference)

    kept_pred = np.flatnonzero(pred_areas >= min_lesion_px) + 1
    kept_pred_mask = np.isin(pred_labels, kept_pred)
    kept_ref = np.flatnonzero(ref_areas >= min_lesion_px) + 1

    hit_ref = np.unique(ref_labels[kept_pred_mask])
    detected = int(np.isin(kept_ref, hit_ref).sum())
    true_pred = np.unique(pred_labels[reference])
    false_lesions = int((~np.isin(kept_pred, true_pred)).sum())

    return {
        'ref_px': int(reference.sum()),
        'pred_px': int(predicted.sum()),
        'tp': int((predicted & reference).sum()),
        'fp': int((predicted & ~reference).sum()),
        'fn': int((~predicted & reference).sum()),
        'max_prob': float(np.max(probs)),
        'largest_pred_component_px': int(pred_areas.max()) if len(pred_areas)
        else 0,
        'pred_lesions': int(len(kept_pred)),
        'ref_lesions': int(len(kept_ref)),
        'ref_lesions_detected': detected,
        'fp_lesions': false_lesions,
    }


def wilson_interval(successes, total, z=1.959964):
    """Wilson score interval for a binomial proportion."""
    if total == 0:
        return [float('nan'), float('nan')]
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    centre = (proportion + z ** 2 / (2 * total)) / denominator
    half = (z * math.sqrt(proportion * (1 - proportion) / total
                          + z ** 2 / (4 * total ** 2)) / denominator)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def auroc(scores, labels):
    """Area under the ROC curve via average ranks (ties count one half)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float('nan')
    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    return float((ranks[labels == 1].sum()
                  - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def json_safe(value):
    """Replace NaN with None so the written file is strict JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def bscan_metrics(records, min_lesion_px):
    """Scan-level confusion counts for one minimum-component threshold."""
    labels = np.array([r['label'] for r in records], dtype=int)
    calls = np.array([r['largest_pred_component_px'] >= min_lesion_px
                      for r in records], dtype=int)
    tp = int(((calls == 1) & (labels == 1)).sum())
    fp = int(((calls == 1) & (labels == 0)).sum())
    tn = int(((calls == 0) & (labels == 0)).sum())
    fn = int(((calls == 0) & (labels == 1)).sum())
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        'min_lesion_px': min_lesion_px,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'sensitivity': sensitivity,
        'sensitivity_ci95': wilson_interval(tp, tp + fn),
        'specificity': specificity,
        'specificity_ci95': wilson_interval(tn, tn + fp),
        'accuracy': _ratio(tp + tn, len(records)),
        'balanced_accuracy': (sensitivity + specificity) / 2,
        'precision': _ratio(tp, tp + fp),
        'f1': _ratio(2 * tp, 2 * tp + fp + fn),
    }


def summarize(records, min_lesion_px=25):
    """Aggregate per-image records into scan, lesion and pixel metrics."""
    present = [r for r in records if r['label'] == 1]
    absent = [r for r in records if r['label'] == 0]
    labels = [r['label'] for r in records]

    tp = sum(r['tp'] for r in records)
    fp = sum(r['fp'] for r in records)
    fn = sum(r['fn'] for r in records)
    dice = [_ratio(2 * r['tp'], 2 * r['tp'] + r['fp'] + r['fn'])
            for r in present]
    ref_lesions = sum(r['ref_lesions'] for r in records)
    detected = sum(r['ref_lesions_detected'] for r in records)

    return {
        'images': len(records),
        'present': len(present),
        'absent': len(absent),
        'bscan': bscan_metrics(records, min_lesion_px),
        'bscan_threshold_sweep': [
            {key: value for key, value in bscan_metrics(records, px).items()
             if key in ('min_lesion_px', 'sensitivity', 'specificity',
                        'balanced_accuracy')}
            for px in BSCAN_SWEEP_PX],
        'auroc_largest_component': auroc(
            [r['largest_pred_component_px'] for r in records], labels),
        'auroc_max_prob': auroc([r['max_prob'] for r in records], labels),
        'lesion': {
            'min_lesion_px': min_lesion_px,
            'reference_lesions': ref_lesions,
            'detected': detected,
            'recall': _ratio(detected, ref_lesions),
            'false_positive_lesions': sum(r['fp_lesions'] for r in records),
            'false_positive_lesions_per_scan': _ratio(
                sum(r['fp_lesions'] for r in records), len(records)),
            'false_positive_lesions_per_absent_scan': _ratio(
                sum(r['fp_lesions'] for r in absent), len(absent)),
        },
        'pixel': {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': _ratio(tp, tp + fp),
            'recall': _ratio(tp, tp + fn),
            'f1': _ratio(2 * tp, 2 * tp + fp + fn),
            'iou': _ratio(tp, tp + fp + fn),
            'mean_dice_present_scans': (float(np.mean(dice)) if dice
                                        else float('nan')),
            'false_positive_px_per_absent_scan': _ratio(
                sum(r['fp'] for r in absent), len(absent)),
        },
    }


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def model_geometry(model_type, patch_size=572):
    """Return ``(in_w, out_w)`` exactly as the Trainer derives them."""
    if model_type in ('retfound', 'retfound_rfa'):
        return 224, 224
    return patch_size, patch_size - 72


def resolve_checkpoint(project_dir, checkpoint):
    model_dir = Path(project_dir) / 'models'
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_file():
            path = model_dir / checkpoint
        if not path.is_file():
            raise SystemExit(f'Checkpoint not found: {checkpoint}')
        return path
    checkpoints = sorted(model_dir.glob('*.pkl'))
    if not checkpoints:
        raise SystemExit(f'No checkpoints in {model_dir}')
    return checkpoints[-1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--dataset', type=Path, default=None,
                        help='curated dataset (default: the one recorded in '
                             'the project\'s offline_experiment.json)')
    parser.add_argument('--split', default='val',
                        choices=('val', 'train', 'all'))
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint file name or path (default: latest, '
                             'i.e. validation-best)')
    parser.add_argument('--gpu', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-lesion-px', type=int, default=25,
                        help='smallest connected component treated as a '
                             'lesion (default 25)')
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--no-save-predictions', action='store_true')
    parser.add_argument(
        '--write-project-segmentations', action='store_true',
        help='also write predictions into <project>/segmentations so the '
             'painter displays this checkpoint without a running trainer')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    project_dir = args.project.resolve()
    project_files = sorted(project_dir.glob('*.seg_proj'))
    if len(project_files) != 1:
        raise SystemExit(f'Expected one .seg_proj file in {project_dir}')
    with open(project_files[0], encoding='utf-8') as stream:
        settings = json.load(stream)
    model_type = settings.get('model_type') or 'unet'

    dataset = args.dataset
    if dataset is None:
        experiment_path = project_dir / 'offline_experiment.json'
        if not experiment_path.is_file():
            raise SystemExit('Pass --dataset: the project has no '
                             'offline_experiment.json')
        with open(experiment_path, encoding='utf-8') as stream:
            dataset = Path(json.load(stream)['dataset'])
    dataset = dataset.resolve()
    ground_truth = dataset / 'ground_truth'

    labels = read_rows(ground_truth / 'bscan_labels.csv')
    reviewed = {row['filename']: row for row in
                read_rows(ground_truth / 'reviewed_mask_index.csv')}
    rows = [row for row in labels
            if args.split == 'all' or row['split'] == args.split]
    if not rows:
        raise SystemExit(f'No images in split {args.split}')

    checkpoint = resolve_checkpoint(project_dir, args.checkpoint)
    out_dir = args.out or (project_dir / 'evaluation'
                           / f'{args.split}__{checkpoint.stem}')
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(SRC_DIR))
    import im_utils  # pylint: disable=import-outside-toplevel
    import model_utils  # pylint: disable=import-outside-toplevel

    in_w, out_w = model_geometry(model_type)
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        with contextlib.redirect_stdout(devnull):
            model = model_utils.load_model(str(checkpoint),
                                           model_type=model_type)

    records = []
    skipped = 0
    for index, row in enumerate(rows, start=1):
        filename = row['filename']
        stem = Path(filename).stem
        image = im_utils.load_image(str(dataset / 'images' / filename))
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                probs = model_utils.ensemble_foreground_probs(
                    [model], image, args.batch_size, in_w, out_w)
        predicted = probs > args.threshold
        if not args.no_save_predictions:
            write_segmentation(out_dir / 'predictions' / (stem + '.png'),
                               predicted)
        if args.write_project_segmentations:
            write_segmentation(
                project_dir / 'segmentations' / (stem + '.png'), predicted)

        review = reviewed.get(filename, {})
        if (row['decision'] not in ('present', 'absent')
                or review.get('status') != 'available'):
            skipped += 1
            continue
        reference = read_binary_mask(ground_truth / review['mask'])
        if reference.shape != probs.shape:
            raise SystemExit('A reference mask and its image differ in size')
        record = image_record(probs, reference, args.threshold,
                              args.min_lesion_px)
        record.update(filename=filename, patient_id=row.get('patient_id', ''),
                      split=row['split'], decision=row['decision'],
                      label=int(row['decision'] == 'present'))
        records.append(record)
        if index % 50 == 0 or index == len(rows):
            print(f'  evaluated {index}/{len(rows)}', flush=True)

    if not records:
        raise SystemExit('No scan in this split has a definite reference')
    summary = summarize(records, args.min_lesion_px)
    summary.update(
        project=project_dir.name, model_type=model_type,
        checkpoint=checkpoint.name, split=args.split,
        threshold=args.threshold, dataset=str(dataset),
        excluded_without_definite_reference=skipped)
    summary = json_safe(summary)
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    identity = ['filename', 'patient_id', 'split', 'decision', 'label']
    fields = identity + [key for key in records[0] if key not in identity]
    with open(out_dir / 'per_image.csv', 'w', encoding='utf-8',
              newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2))
    print(f'Wrote {out_dir}')


if __name__ == '__main__':
    main()
