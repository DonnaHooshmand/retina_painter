"""Paired comparison of two evaluations produced by ``evaluate_project.py``.

Both evaluations must cover exactly the same scans. Differences are reported
as B minus A with 95% intervals from a patient-level bootstrap: scans from one
patient are correlated, so whole patients are resampled rather than individual
scans. Scan-level calls are additionally compared with an exact McNemar test.

An interval that contains zero means this dataset cannot distinguish the two
models on that metric; it does not show they are equivalent.

    python compare_evaluations.py --a <project A>/evaluation/val__... \\
                                  --b <project B>/evaluation/val__...

CLI output is aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_project import json_safe, summarize

INT_FIELDS = ('label', 'ref_px', 'pred_px', 'tp', 'fp', 'fn',
              'largest_pred_component_px', 'pred_lesions', 'ref_lesions',
              'ref_lesions_detected', 'fp_lesions')


def load_records(evaluation_dir):
    with open(Path(evaluation_dir) / 'per_image.csv', encoding='utf-8',
              newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in INT_FIELDS:
            row[field] = int(row[field])
        row['max_prob'] = float(row['max_prob'])
    return {row['filename']: row for row in rows}


def headline(records, min_lesion_px):
    """The scalar metrics compared between two models."""
    summary = summarize(records, min_lesion_px)
    return {
        'scan sensitivity': summary['bscan']['sensitivity'],
        'scan specificity': summary['bscan']['specificity'],
        'scan balanced accuracy': summary['bscan']['balanced_accuracy'],
        'scan AUROC (largest component)': summary['auroc_largest_component'],
        'lesion recall': summary['lesion']['recall'],
        'false-positive lesions / absent scan':
            summary['lesion']['false_positive_lesions_per_absent_scan'],
        'pixel F1 (pooled)': summary['pixel']['f1'],
        'pixel precision': summary['pixel']['precision'],
        'pixel recall': summary['pixel']['recall'],
        'mean Dice on present scans':
            summary['pixel']['mean_dice_present_scans'],
    }


def mcnemar_exact(records_a, records_b, min_lesion_px):
    """Two-sided exact McNemar test on scan-level correctness."""
    def correct(record):
        call = record['largest_pred_component_px'] >= min_lesion_px
        return int(call) == record['label']
    only_a = sum(correct(a) and not correct(b)
                 for a, b in zip(records_a, records_b))
    only_b = sum(correct(b) and not correct(a)
                 for a, b in zip(records_a, records_b))
    discordant = only_a + only_b
    if discordant == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(only_a, only_b) + 1)) / 2 ** discordant
    return only_a, only_b, min(1.0, 2 * tail)


def paired_bootstrap(records_a, records_b, min_lesion_px, resamples=2000,
                     seed=0):
    """Patient-level bootstrap intervals for B minus A on every metric."""
    by_patient = defaultdict(list)
    for index, record in enumerate(records_a):
        by_patient[record.get('patient_id') or record['filename']].append(
            index)
    groups = list(by_patient.values())
    rng = np.random.default_rng(seed)
    differences = defaultdict(list)
    for _ in range(resamples):
        chosen = [index
                  for group in rng.integers(0, len(groups), len(groups))
                  for index in groups[group]]
        sample_a = headline([records_a[i] for i in chosen], min_lesion_px)
        sample_b = headline([records_b[i] for i in chosen], min_lesion_px)
        for name in sample_a:
            differences[name].append(sample_b[name] - sample_a[name])
    intervals = {}
    for name, values in differences.items():
        values = np.array(values, dtype=float)
        values = values[np.isfinite(values)]
        intervals[name] = ([float(np.percentile(values, 2.5)),
                            float(np.percentile(values, 97.5))]
                           if len(values) else [float('nan')] * 2)
    return intervals, len(groups)


def compare(dir_a, dir_b, min_lesion_px=25, resamples=2000, seed=0):
    loaded_a, loaded_b = load_records(dir_a), load_records(dir_b)
    if set(loaded_a) != set(loaded_b):
        raise ValueError('The two evaluations cover different scans')
    names = sorted(loaded_a)
    records_a = [loaded_a[name] for name in names]
    records_b = [loaded_b[name] for name in names]
    metrics_a = headline(records_a, min_lesion_px)
    metrics_b = headline(records_b, min_lesion_px)
    intervals, patients = paired_bootstrap(
        records_a, records_b, min_lesion_px, resamples, seed)
    only_a, only_b, p_value = mcnemar_exact(
        records_a, records_b, min_lesion_px)
    return {
        'scans': len(names),
        'patients': patients,
        'min_lesion_px': min_lesion_px,
        'metrics': {
            name: {'a': metrics_a[name], 'b': metrics_b[name],
                   'b_minus_a': metrics_b[name] - metrics_a[name],
                   'ci95': intervals[name]}
            for name in metrics_a},
        'mcnemar': {'scans_only_a_correct': only_a,
                    'scans_only_b_correct': only_b, 'p_value': p_value},
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--a', required=True, type=Path)
    parser.add_argument('--b', required=True, type=Path)
    parser.add_argument('--min-lesion-px', type=int, default=25)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--out', type=Path, default=None,
                        help='optional path for the comparison JSON')
    args = parser.parse_args()

    result = compare(args.a, args.b, args.min_lesion_px, args.resamples)
    for key, directory in (('a', args.a), ('b', args.b)):
        with open(directory / 'metrics.json', encoding='utf-8') as stream:
            meta = json.load(stream)
        result[key] = {field: meta.get(field) for field in (
            'project', 'model_type', 'checkpoint', 'split')}

    print(f"A = {result['a']['project']} ({result['a']['model_type']})")
    print(f"B = {result['b']['project']} ({result['b']['model_type']})")
    print(f"{result['scans']} scans from {result['patients']} patients, "
          f"split={result['a']['split']}\n")
    print(f"{'metric':38s} {'A':>8s} {'B':>8s} {'B-A':>8s}  95% CI (B-A)")
    for name, row in result['metrics'].items():
        low, high = row['ci95']
        print(f"{name:38s} {row['a']:8.3f} {row['b']:8.3f} "
              f"{row['b_minus_a']:+8.3f}  [{low:+.3f}, {high:+.3f}]")
    test = result['mcnemar']
    print(f"\nScan-level calls: only A correct on "
          f"{test['scans_only_a_correct']}, only B correct on "
          f"{test['scans_only_b_correct']} (exact McNemar p = "
          f"{test['p_value']:.3f})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as stream:
            json.dump(json_safe(result), stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
