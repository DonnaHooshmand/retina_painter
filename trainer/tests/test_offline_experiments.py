"""Tests for the offline experiment tools in ``trainer/experiments``."""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
experiments_dir = os.path.join(os.path.dirname(test_dir), 'experiments')
sys.path.insert(0, src_dir)
sys.path.insert(0, experiments_dir)

import model_utils
import evaluate_project
import mask_io
import prepare_offline_project
import run_offline_training
from evaluate_project import (auroc, bscan_metrics, image_record, summarize,
                              wilson_interval)
from prepare_offline_project import prepare_offline_project as prepare

SHAPE = (48, 64)


def _save_grey(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _blob(top, left, size):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[top:top + size, left:left + size] = True
    return mask


def _make_dataset(root):
    """Tiny curated dataset: 2 train patients, 1 val patient, 1 uncertain."""
    dataset = root / 'sync' / 'curated_datasets' / 'tiny'
    (root / 'sync' / 'datasets').mkdir(parents=True)
    scans = [
        # filename, patient, split, decision, reviewed mask, strokes
        ('a1.png', 'A', 'train', 'present', _blob(10, 10, 5), True),
        ('a2.png', 'A', 'train', 'absent', np.zeros(SHAPE, bool), True),
        ('b1.png', 'B', 'train', 'present', _blob(0, 0, 40), True),
        ('b2.png', 'B', 'train', 'absent', np.zeros(SHAPE, bool), False),
        ('c1.png', 'C', 'val', 'present', _blob(20, 30, 6), True),
        ('c2.png', 'C', 'val', 'absent', np.zeros(SHAPE, bool), True),
        ('c3.png', 'C', 'val', 'uncertain', None, False),
    ]
    labels, reviewed = [], []
    for filename, patient, split, decision, mask, strokes in scans:
        stem = filename[:-4]
        image_path = dataset / 'images' / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full(SHAPE + (3,), 90, np.uint8)).save(image_path)
        labels.append(dict(filename=filename, patient_id=patient, split=split,
                           decision=decision))
        row = dict(filename=filename, split=split, decision=decision,
                   mask='', status='no_reviewed_mask')
        if mask is not None:
            row.update(mask=f'reviewed_masks/{filename}', status='available')
            _save_grey(dataset / 'ground_truth' / row['mask'], mask)
        reviewed.append(row)
        if strokes:
            foreground = np.zeros(SHAPE, bool)
            if decision == 'present':
                foreground[np.argwhere(mask)[0][0], np.argwhere(mask)[0][1]] = True
            background = np.zeros(SHAPE, bool)
            background[45:47, 0:20] = True
            mask_io.write_annotation(
                dataset / 'annotations' / split / filename,
                foreground, background)
    for name, rows in (('bscan_labels.csv', labels),
                       ('reviewed_mask_index.csv', reviewed)):
        with open(dataset / 'ground_truth' / name, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return dataset, root / 'sync'


# -- mask encodings ---------------------------------------------------------

def test_read_binary_mask_decodes_both_reference_encodings(tmp_path):
    mask = _blob(5, 5, 4)
    _save_grey(tmp_path / 'grey.png', mask)
    mask_io.write_segmentation(tmp_path / 'painter.png', mask)

    assert np.array_equal(mask_io.read_binary_mask(tmp_path / 'grey.png'), mask)
    assert np.array_equal(
        mask_io.read_binary_mask(tmp_path / 'painter.png'), mask)
    # The trap: the painter's cyan encoding has an empty red channel, so a
    # naive "channel 0 > 0" read sees no foreground at all.
    naive = np.asarray(Image.open(tmp_path / 'painter.png'))[:, :, 0] > 0
    assert not naive.any()


def test_written_annotation_matches_trainer_channel_decoding(tmp_path):
    foreground = _blob(5, 5, 4)
    mask_io.write_annotation(tmp_path / 'a.png', foreground, ~foreground)
    raw = np.asarray(Image.open(tmp_path / 'a.png'))
    # trainer/datasets.py and model_utils.get_val_metrics read these channels
    assert np.array_equal(raw[:, :, 0].astype(bool), foreground)
    assert np.array_equal(raw[:, :, 1].astype(bool), ~foreground)
    assert mask_io.read_annotation(tmp_path / 'a.png')[0].sum() == 16

    with pytest.raises(ValueError):
        mask_io.write_annotation(tmp_path / 'b.png', foreground, foreground)


# -- project preparation ----------------------------------------------------

def test_sparse_project_copies_strokes_and_keeps_patient_split(tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    summary = prepare(dataset, sync_dir, 'proj', model_type='retfound_rfa',
                      seed=3)
    project = sync_dir / 'projects' / 'proj'

    assert sorted(p.name for p in (project / 'annotations/train').iterdir()) \
        == ['a1.png', 'a2.png', 'b1.png']
    assert sorted(p.name for p in (project / 'annotations/val').iterdir()) \
        == ['c1.png', 'c2.png']
    source = (dataset / 'annotations/train/a1.png').read_bytes()
    assert (project / 'annotations/train/a1.png').read_bytes() == source
    assert summary['annotation_counts']['train_foreground_files'] == 2

    settings = json.loads((project / 'proj.seg_proj').read_text())
    assert settings['model_type'] == 'retfound_rfa'
    assert settings['training_seed'] == 3
    assert sorted(settings['file_names']) == sorted(
        f'{s}.png' for s in ('a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'c3'))
    assert sorted(settings['val_file_names']) == ['c1.png', 'c2.png', 'c3.png']
    assert set(settings['train_file_names']).isdisjoint(
        settings['val_file_names'])
    # The painter resolves the dataset relative to <sync>/datasets.
    resolved = (sync_dir / 'datasets' / settings['dataset']).resolve()
    assert resolved == (dataset / 'images').resolve()

    loaded, loaded_sync, images = run_offline_training.load_project(project)
    assert loaded == settings
    assert loaded_sync == sync_dir.resolve()
    assert images == resolved


def test_prepare_refuses_existing_project_and_patient_leak(tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    prepare(dataset, sync_dir, 'proj')
    with pytest.raises(ValueError, match='already exists'):
        prepare(dataset, sync_dir, 'proj')

    labels = dataset / 'ground_truth' / 'bscan_labels.csv'
    labels.write_text(labels.read_text().replace(
        'c2.png,C,val', 'c2.png,A,val'))
    with pytest.raises(ValueError, match='patient-separated'):
        prepare(dataset, sync_dir, 'leaky')
    assert not (sync_dir / 'projects' / 'leaky').exists()


def test_sparse_absent_project_makes_absent_scans_whole_scan_background(
        tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    summary = prepare(dataset, sync_dir, 'proj', supervision='sparse_absent')
    annotations = sync_dir / 'projects' / 'proj' / 'annotations'

    # Absent scans, with or without strokes, are background everywhere.
    for split, name in (('train', 'a2'), ('train', 'b2'), ('val', 'c2')):
        foreground, background = mask_io.read_annotation(
            annotations / split / f'{name}.png')
        assert background.shape == SHAPE
        assert background.all() and not foreground.any()
    # Present scans keep the clinician's strokes byte for byte.
    for split, name in (('train', 'a1'), ('train', 'b1'), ('val', 'c1')):
        assert (annotations / split / f'{name}.png').read_bytes() == (
            dataset / 'annotations' / split / f'{name}.png').read_bytes()
    # Uncertain scans are never supervised.
    assert not (annotations / 'val' / 'c3.png').exists()

    counts = summary['annotation_counts']
    assert summary['supervision'] == 'sparse_absent'
    assert counts['train_absent_whole_scan_files'] == 2
    assert counts['train_absent_scans_added'] == 1  # b2 had no strokes
    assert counts['val_absent_whole_scan_files'] == 1
    assert counts['train_files'] == 4 and counts['val_files'] == 2
    assert counts['train_foreground_files'] == 2

    # Validation-only variant: training keeps plain strokes.
    prepare(dataset, sync_dir, 'val_only', supervision='sparse_absent_val')
    val_only = sync_dir / 'projects' / 'val_only' / 'annotations'
    assert (val_only / 'train' / 'a2.png').read_bytes() == (
        dataset / 'annotations' / 'train' / 'a2.png').read_bytes()
    assert not (val_only / 'train' / 'b2.png').exists()
    assert mask_io.read_annotation(val_only / 'val' / 'c2.png')[1].all()

    # A red stroke on a scan decided absent is a contradiction, not a label.
    mask_io.write_annotation(
        dataset / 'annotations' / 'train' / 'a2.png',
        _blob(5, 5, 3), np.zeros(SHAPE, bool))
    with pytest.raises(ValueError, match='absent has foreground strokes'):
        prepare(dataset, sync_dir, 'contradiction',
                supervision='sparse_absent')


def test_dense_project_supervises_all_pixels_but_distrusts_inflated_masks(
        tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    summary = prepare(dataset, sync_dir, 'dense', supervision='dense',
                      dense_max_foreground_fraction=0.05)
    project = sync_dir / 'projects' / 'dense'

    # Reliable positive: every pixel supervised, foreground == reviewed mask.
    foreground, background = mask_io.read_annotation(
        project / 'annotations/train/a1.png')
    assert np.array_equal(foreground, _blob(10, 10, 5))
    assert np.array_equal(background, ~foreground)
    # Absent scan without strokes becomes a fully supervised negative.
    foreground, background = mask_io.read_annotation(
        project / 'annotations/train/b2.png')
    assert not foreground.any() and background.all()
    # Inflated positive mask (covers half the scan) is not trusted: the scan
    # keeps only the clinician's own strokes.
    assert (project / 'annotations/train/b1.png').read_bytes() == (
        dataset / 'annotations/train/b1.png').read_bytes()
    # Uncertain scans are never supervised.
    assert not (project / 'annotations/val/c3.png').exists()

    counts = summary['annotation_counts']
    assert counts['train_dense_files'] == 3
    assert counts['train_sparse_fallback_files'] == 1
    assert counts['val_dense_files'] == 2
    assert counts['val_without_reviewed_mask'] == 1


# -- metrics ----------------------------------------------------------------

def test_image_record_counts_pixels_and_lesions():
    reference = _blob(10, 10, 6) | _blob(30, 40, 6)      # two 36 px lesions
    probs = np.zeros(SHAPE, dtype=np.float32)
    probs[10:16, 10:14] = 0.9      # 24 px inside lesion 1
    probs[10:16, 14:18] = 0.9      # 12 px inside + 12 px spill -> one 48 px blob
    probs[0:6, 50:56] = 0.8        # 36 px false-positive blob
    probs[40, 0] = 0.99            # 1 px speck, below min lesion size

    record = image_record(probs, reference, threshold=0.5, min_lesion_px=25)

    assert record['ref_px'] == 72
    assert record['tp'] == 36
    assert record['fp'] == 12 + 36 + 1
    assert record['fn'] == 36
    assert record['largest_pred_component_px'] == 48
    assert record['pred_lesions'] == 2
    assert record['ref_lesions'] == 2
    assert record['ref_lesions_detected'] == 1
    assert record['fp_lesions'] == 1
    assert record['max_prob'] == pytest.approx(0.99)


def test_image_record_handles_empty_prediction_and_reference():
    record = image_record(np.zeros(SHAPE), np.zeros(SHAPE, bool))
    assert record['largest_pred_component_px'] == 0
    assert record['tp'] == record['fp'] == record['fn'] == 0
    assert record['fp_lesions'] == record['ref_lesions'] == 0


def _record(label, largest, tp=0, fp=0, fn=0, **extra):
    record = dict(label=label, largest_pred_component_px=largest, tp=tp,
                  fp=fp, fn=fn, max_prob=min(1.0, largest / 100),
                  ref_lesions=label, ref_lesions_detected=0, fp_lesions=0)
    record.update(extra)
    return record


def test_summarize_reports_scan_lesion_and_pixel_views():
    records = [
        _record(1, 300, tp=80, fp=20, fn=20, ref_lesions_detected=1),
        _record(1, 0, fn=50),                      # missed positive
        _record(0, 0),
        _record(0, 10, fp=10),                     # speck: not a positive call
        _record(0, 60, fp=60, fp_lesions=1),       # false-positive scan
    ]
    summary = summarize(records, min_lesion_px=25)

    scan = summary['bscan']
    assert (scan['tp'], scan['fn'], scan['tn'], scan['fp']) == (1, 1, 2, 1)
    assert scan['sensitivity'] == pytest.approx(0.5)
    assert scan['specificity'] == pytest.approx(2 / 3)
    assert summary['lesion']['recall'] == pytest.approx(0.5)
    assert summary['lesion']['false_positive_lesions_per_absent_scan'] == \
        pytest.approx(1 / 3)
    pixel = summary['pixel']
    assert pixel['precision'] == pytest.approx(80 / 170)
    assert pixel['recall'] == pytest.approx(80 / 150)
    assert pixel['mean_dice_present_scans'] == pytest.approx((0.8 + 0.0) / 2)
    assert pixel['false_positive_px_per_absent_scan'] == pytest.approx(70 / 3)
    # A 1 px threshold turns the speck scan into a second false positive.
    assert bscan_metrics(records, 1)['fp'] == 2
    # 300 beats all three negatives; 0 ties one negative and loses to two.
    assert summary['auroc_largest_component'] == pytest.approx(3.5 / 6)


def test_auroc_and_wilson_interval():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert math.isnan(auroc([0.1, 0.2], [1, 1]))

    low, high = wilson_interval(19, 19)
    assert high == pytest.approx(1.0) and 0.82 < low < 0.84
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)


def test_evaluation_probabilities_are_the_painter_overlay_probabilities(
        monkeypatch):
    rng = np.random.default_rng(0)
    probability_map = rng.random((8, 8)).astype(np.float32)

    def fake_unet_segment(cnn, image, bs, in_w, out_w, threshold):
        return probability_map.copy()

    monkeypatch.setattr(model_utils, 'unet_segment', fake_unet_segment)
    model = torch.nn.Linear(2, 2)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    probs = model_utils.ensemble_foreground_probs([model], image, 1, 8, 8)
    overlay = model_utils.ensemble_segment_models([model], image, 1, 8, 8)

    assert np.allclose(probs, (probability_map + np.fliplr(probability_map)) / 2)
    assert np.array_equal(overlay, (probs > 0.5).astype(int))


def _write_evaluation(directory, largest_by_scan):
    directory.mkdir()
    rows = []
    for index, (label, largest) in enumerate(largest_by_scan):
        row = _record(label, largest, tp=10 * label, fn=5 * label,
                      ref_lesions_detected=int(label and largest >= 25))
        row.update(filename=f's{index}.png', patient_id=f'p{index // 2}',
                   ref_px=15 * label, pred_px=largest, pred_lesions=0)
        rows.append(row)
    with open(directory / 'per_image.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paired_comparison_reports_difference_and_mcnemar(tmp_path):
    import compare_evaluations
    # A misses three positives that B finds; both are right on negatives.
    scans_a = [(1, 0), (1, 0), (1, 0), (1, 90), (0, 0), (0, 0), (0, 0), (0, 0)]
    scans_b = [(1, 80), (1, 80), (1, 80), (1, 90), (0, 0), (0, 0), (0, 0),
               (0, 0)]
    _write_evaluation(tmp_path / 'a', scans_a)
    _write_evaluation(tmp_path / 'b', scans_b)

    result = compare_evaluations.compare(
        tmp_path / 'a', tmp_path / 'b', resamples=200)

    assert result['scans'] == 8 and result['patients'] == 4
    sensitivity = result['metrics']['scan sensitivity']
    assert sensitivity['a'] == pytest.approx(0.25)
    assert sensitivity['b'] == pytest.approx(1.0)
    assert sensitivity['b_minus_a'] == pytest.approx(0.75)
    low, high = sensitivity['ci95']
    assert 0 < low <= 0.75 <= high <= 1.0
    assert result['metrics']['scan specificity']['b_minus_a'] == 0
    test = result['mcnemar']
    assert (test['scans_only_a_correct'], test['scans_only_b_correct']) == (0, 3)
    assert test['p_value'] == pytest.approx(0.25)

    _write_evaluation(tmp_path / 'c', scans_a[:-1])
    with pytest.raises(ValueError, match='different scans'):
        compare_evaluations.compare(tmp_path / 'a', tmp_path / 'c')


def test_model_geometry_matches_trainer():
    assert evaluate_project.model_geometry('retfound_rfa') == (224, 224)
    assert evaluate_project.model_geometry('retfound') == (224, 224)
    assert evaluate_project.model_geometry('unet') == (572, 500)


# -- patient-level cross-validation -----------------------------------------

def _cohort(patients=40, positive_every=3, scans=3):
    """Label rows: every third patient has one positive scan."""
    rows = []
    for patient in range(patients):
        for scan in range(scans):
            positive = patient % positive_every == 0 and scan == 0
            rows.append(dict(filename=f'p{patient:02d}_{scan}.png',
                             patient_id=f'p{patient:02d}', split='train',
                             decision='present' if positive else 'absent'))
    return rows


def test_patient_folds_keep_patients_whole_and_spread_positives():
    from prepare_offline_project import fold_routes, patient_folds

    rows = _cohort()
    folds = patient_folds(rows, 4, seed=0)
    assert folds == patient_folds(list(reversed(rows)), 4, seed=0)
    assert folds != patient_folds(rows, 4, seed=1)
    assert set(folds.values()) == {0, 1, 2, 3}
    # 14 positive patients over 4 folds: no fold has fewer than 3 or more than 4.
    positive = {row['patient_id'] for row in rows
                if row['decision'] == 'present'}
    per_fold = [sum(folds[p] == k for p in positive) for k in range(4)]
    assert sum(per_fold) == 14 and min(per_fold) >= 3 and max(per_fold) <= 4

    held_out = set()
    for fold in range(4):
        route = fold_routes(rows, 4, fold, seed=0)
        roles = {}
        for row in rows:
            roles.setdefault(row['patient_id'], set()).add(
                route[row['filename']])
        # No patient in two roles; every role is used.
        assert all(len(found) == 1 for found in roles.values())
        assert {'train', 'val', 'test'} == set(route.values())
        test = {name for name, role in route.items() if role == 'test'}
        assert not held_out & test
        held_out |= test
        # Validation takes about one sixth of what is left.
        val = sum(role == 'val' for role in route.values())
        train = sum(role == 'train' for role in route.values())
        assert 0.1 < val / (val + train) < 0.25
    # Across the folds every scan is held out exactly once.
    assert held_out == {row['filename'] for row in rows}

    with pytest.raises(ValueError, match='between 0 and 3'):
        fold_routes(rows, 4, 4)


def test_fold_project_holds_out_its_patients(tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    # Three patients: with three folds each is held out once.
    held_out = []
    for fold in range(3):
        summary = prepare(dataset, sync_dir, f'cv_fold{fold}', model_type='unet',
                          supervision='sparse_absent', folds=3, fold=fold)
        project = sync_dir / 'projects' / f'cv_fold{fold}'
        settings, _sync, _images = run_offline_training.load_project(project)
        test = settings['test_file_names']
        patients = {name[0] for name in test}
        assert len(patients) == 1 and settings['split_method'] == 'patient_fold_v1'
        held_out.append(patients.pop())
        assert summary['fold'] == fold and summary['folds'] == 3
        assert sum(summary['split_images'].values()) == 7
        # Held-out scans are never annotated; the others are where routed.
        annotated = {path.name for path in
                     (project / 'annotations').rglob('*.png')}
        assert not annotated & set(test)
        for role in ('train', 'val'):
            names = {path.name for path in
                     (project / 'annotations' / role).glob('*.png')}
            assert names <= set(settings[f'{role}_file_names'])
        assert not set(settings['train_file_names']) & set(
            settings['val_file_names'])
    assert sorted(held_out) == ['a', 'b', 'c']

    with pytest.raises(ValueError, match='both folds and fold'):
        prepare(dataset, sync_dir, 'half', folds=3)


def test_blob_score_and_calibration():
    import score_cross_validation as cv

    probs = np.zeros(SHAPE)
    probs[5:11, 5:11] = 0.62      # 36 px blob
    probs[30:33, 30:33] = 0.95    # 9 px: too small to count
    assert cv.blob_score(probs, min_px=25) == 0.61
    assert cv.blob_score(probs, min_px=5) == 0.94
    assert cv.blob_score(np.zeros(SHAPE)) == 0.0
    # Two fragments that only join into a blob at a lower threshold.
    joined = np.zeros(SHAPE)
    joined[0:4, 0:4] = 0.9
    joined[0:4, 4:8] = 0.4
    assert cv.blob_score(joined, min_px=25) == 0.39

    negatives = [0.0] * 8 + [0.3, 0.7]
    # 0.31 leaves 9 of 10 unflagged; 0.3 would flag the scan scoring 0.3.
    assert cv.calibrate_threshold(negatives, 0.9) == 0.31
    assert cv.calibrate_threshold(negatives, 1.0) == 0.71
    assert cv.calibrate_threshold([0.0, 0.0], 0.9) == 0.01
    with pytest.raises(ValueError, match='absent'):
        cv.calibrate_threshold([], 0.9)


def _held_out(name, patient, decision, score, flagged, lesions=0, found=0,
              fp_blobs=0):
    return dict(filename=name, patient_id=patient, role='test',
                decision=decision, blob_score=score, flagged=flagged,
                lesions=lesions, lesions_found=found, fp_blobs=fp_blobs)


def test_cross_validation_summary_counts_blobs_and_compares_runs():
    import score_cross_validation as cv

    baseline = [
        _held_out('a1', 'A', 'present', 0.9, 1, lesions=1, found=1),
        _held_out('a2', 'A', 'present', 0.2, 0, lesions=2, found=0),
        _held_out('b1', 'B', 'absent', 0.1, 0),
        _held_out('b2', 'B', 'absent', 0.6, 1, fp_blobs=2),
        _held_out('c1', 'C', 'absent', 0.0, 0),
        _held_out('c2', 'C', 'present', 0.5, 1, lesions=1, found=1),
    ]
    summary = cv.summarize(baseline)
    assert summary['positive_scans_flagged'] == 2 and summary['positive_scans'] == 3
    assert summary['negative_scans_cleared'] == 2
    assert summary['sensitivity'] == pytest.approx(2 / 3)
    assert summary['specificity'] == pytest.approx(2 / 3)
    assert summary['lesions_found'] == 2 and summary['lesions'] == 4
    assert summary['fp_blobs_per_negative_scan'] == pytest.approx(2 / 3)
    # 9 positive/negative pairs; only (a2, b2) and (c2, b2) are misordered.
    assert summary['auroc'] == pytest.approx(7 / 9)
    # Rows read back from fold_scores.csv are strings.
    as_text = [{key: str(value) for key, value in row.items()}
               for row in baseline]
    assert cv.summarize(as_text) == summary

    intervals, patients = cv.bootstrap(baseline, resamples=200)
    assert patients == 3
    assert intervals['sensitivity'][0] <= 2 / 3 <= intervals['sensitivity'][1]

    better = [dict(row) for row in baseline]
    better[1].update(blob_score=0.8, flagged=1, lesions_found=2)   # a2 found
    better[3].update(blob_score=0.05, flagged=0, fp_blobs=0)        # b2 cleared
    versus = cv.compare_runs(better, baseline)
    assert versus['sensitivity']['difference'] == pytest.approx(1 / 3)
    assert versus['specificity']['difference'] == pytest.approx(1 / 3)
    assert versus['auroc']['run'] == pytest.approx(1.0)
    assert versus['fp_blobs_per_negative_scan']['difference'] == pytest.approx(
        -2 / 3)
    low, high = versus['sensitivity']['interval']
    assert low >= 0 and high <= 1

    with pytest.raises(ValueError, match='same scans'):
        cv.compare_runs(better[:-1], baseline)


def test_cross_validation_driver_splits_trainer_options():
    import run_cross_validation as driver

    args = driver.parse_args([
        '--dataset', 'd', '--sync-dir', 's', '--name', 'cv', '--folds', '4',
        '--gpus', '0,1', '--max-minutes', '30', '--', '--no-rollback',
        '--loss-type', 'combined_per_tile'])
    assert args.folds == 4 and args.max_minutes == 30.0
    assert args.train_args == ['--no-rollback', '--loss-type',
                               'combined_per_tile']
    command = driver.training_command(Path('p') / 'cv_fold2', '1', 30.0,
                                      args.train_args)
    assert command[-7:] == ['--gpu', '1', '--max-minutes', '30.0',
                            '--no-rollback', '--loss-type',
                            'combined_per_tile']
    assert driver.fold_name('cv', 2) == 'cv_fold2'
    assert driver.parse_args(
        ['--dataset', 'd', '--sync-dir', 's', '--name', 'cv']).train_args == []


# -- error review project ---------------------------------------------------

def _calls(scores):
    """per_image.csv rows from ``{name: (decision, max probability)}``."""
    return [dict(filename=name, decision=decision, max_prob=str(prob))
            for name, (decision, prob) in scores.items()]


def test_error_review_selects_scans_models_agree_are_wrong():
    import prepare_error_review as review

    negatives = {f'n{i}': ('absent', 0.05 * i) for i in range(10)}  # 0 .. 0.45
    model_a = _calls({**negatives, 'easy': ('present', 0.9),
                      'hard': ('present', 0.1), 'odd': ('present', 0.5)})
    # A conservative model: every score lower, same order.
    model_b = _calls({**{k: (d, p / 10) for k, (d, p) in negatives.items()},
                      'easy': ('present', 0.2), 'hard': ('present', 0.001),
                      'odd': ('present', 0.03)})
    model_c = _calls({**negatives, 'easy': ('present', 0.95),
                      'hard': ('present', 0.2), 'odd': ('present', 0.3)})

    # Each model is held at its own 90%-specificity threshold, so the
    # conservative one is not counted as missing everything.
    flagged = review.calls_at_specificity(model_b, 0.9)
    assert flagged['easy'] and not flagged['hard']
    assert sum(flagged[name] for name in negatives) == 1

    decision, flags = review.count_flags([model_a, model_b, model_c], 0.9)
    assert flags['easy'] == 3 and flags['hard'] == 0 and flags['odd'] == 1
    assert flags['n9'] == 3 and flags['n0'] == 0

    scans = review.select_review_scans(decision, flags, models=3)
    assert [(s['filename'], s['group']) for s in scans] == [
        ('hard', 'missed_positive'), ('odd', 'missed_positive'),
        ('easy', 'found_positive'), ('n9', 'flagged_negative')]
    assert scans[0]['models_flagging'] == 0 and scans[0]['models_compared'] == 3

    with pytest.raises(ValueError, match='same scans'):
        review.count_flags([model_a, model_b[:-1]], 0.9)


def test_error_review_project_is_for_looking_not_training(tmp_path):
    import prepare_error_review as review

    dataset, sync_dir = _make_dataset(tmp_path)
    scans = [
        dict(filename='c1.png', clinician_decision='present',
             group='missed_positive', models_flagging=0, models_compared=3),
        dict(filename='b2.png', clinician_decision='absent',
             group='flagged_negative', models_flagging=3, models_compared=3),
    ]
    overlay = _blob(2, 3, 4)
    summary = review.create_review_project(
        dataset, sync_dir, 'review', scans, lambda _name: overlay,
        provenance={'target_specificity': 0.9})
    project = sync_dir / 'projects' / 'review'

    settings, _sync, images_dir = run_offline_training.load_project(project)
    assert images_dir == (dataset / 'images').resolve()
    assert settings['file_names'] == ['c1.png', 'b2.png']
    assert settings['val_file_names'] == ['c1.png', 'b2.png']
    assert settings['train_file_names'] == []
    # Never starts training, and opens on the first scan.
    assert settings['auto_start_training_after_reviews'] == 0
    assert settings['image_index'] == 0
    # The cheapest model, should a running trainer be asked to segment.
    assert settings['model_type'] == 'unet'
    assert not list((project / 'annotations' / 'train').iterdir())

    # The clinician's strokes are shown where they exist (b2 has none).
    assert (project / 'annotations' / 'val' / 'c1.png').read_bytes() == (
        dataset / 'annotations' / 'val' / 'c1.png').read_bytes()
    assert not (project / 'annotations' / 'val' / 'b2.png').exists()
    for stem in ('c1', 'b2'):
        assert np.array_equal(mask_io.read_binary_mask(
            project / 'segmentations' / f'{stem}.png'), overlay)

    with open(project / 'review_list.csv', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['position'], r['filename'], r['group']) for r in rows] == [
        ('1', 'c1.png', 'missed_positive'), ('2', 'b2.png', 'flagged_negative')]
    assert summary['group_counts'] == {
        'missed_positive': 1, 'found_positive': 0, 'flagged_negative': 1}
    assert summary['group_positions'] == {
        'missed_positive': [1, 1], 'flagged_negative': [2, 2]}
    assert summary['scans_with_clinician_strokes'] == 1

    with pytest.raises(ValueError, match='already exists'):
        review.create_review_project(
            dataset, sync_dir, 'review', scans, lambda _name: overlay)


# -- offline trainer --------------------------------------------------------

def test_offline_trainer_leaves_painter_instruction_queue_untouched(tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    prepare(dataset, sync_dir, 'proj', model_type='unet')
    project = sync_dir / 'projects' / 'proj'
    instructions = sync_dir / 'instructions'
    instructions.mkdir()
    pending = instructions / 'segment_123'
    pending.write_text('{"model_type": "retfound"}')

    args = argparse.Namespace(
        patchsize=572, maxworkers=0, maxbatchsize=2, model_type='unet',
        loss_type='auto', max_epochs_without_progress=5)
    trainer = run_offline_training.build_offline_trainer(
        project, sync_dir, args)

    # Construction must not clear the queue (the interactive Trainer does)...
    assert pending.exists()
    # ...and polling must neither consume it nor switch the model type.
    trainer.training = True
    trainer.check_for_instructions()
    assert pending.exists()
    assert trainer.model_type == 'unet' and trainer.training

    trainer.log('hello')
    assert 'hello' in (project / 'logs/offline_server_log.txt').read_text()
    assert not (sync_dir / 'server_log.txt').exists()
    trainer.msg_dir = str(project / 'messages')
    trainer.write_message('Training started')
    assert not list((project / 'messages').iterdir())

    (project / 'STOP').write_text('')
    trainer.check_for_instructions()
    assert trainer.training is False
    assert trainer.stop_reason == 'stop_file'

    # As in the live trainer, automatic rollback is off unless asked for.
    assert trainer.candidate_rollback_patience is None
    # A trainer source from another commit may have no "off": for it the
    # patience is put out of reach instead.
    older = run_offline_training.build_offline_trainer(
        project, sync_dir, args, src_dir=tmp_path / 'older_src')
    assert older.candidate_rollback_patience == sys.maxsize


def test_rollback_is_off_unless_a_patience_is_given():
    parse = run_offline_training.parse_args
    patience = run_offline_training.rollback_patience

    assert patience(parse(['--project', 'p'])) is None
    assert patience(parse(
        ['--project', 'p', '--candidate-rollback-patience', '10'])) == 10
    # --no-rollback is kept for older commands and wins over a patience.
    assert patience(parse(['--project', 'p', '--no-rollback'])) is None
    assert patience(parse(['--project', 'p', '--no-rollback',
                           '--candidate-rollback-patience', '3'])) is None


def test_runner_adapts_to_an_older_trainer_and_takes_a_time_budget():
    def older_init(self, sync_dir=None, patch_size=572, max_workers=12,
                   max_batch_size=12):
        """Signature of the original RootPainter Trainer."""

    def flexible_init(self, **kwargs):
        """Accepts anything."""

    keep = run_offline_training.supported_kwargs
    assert keep(older_init, sync_dir='s', patch_size=572, model_type='unet',
                candidate_rollback_patience=3) == {
                    'sync_dir': 's', 'patch_size': 572}
    assert keep(flexible_init, model_type='unet') == {'model_type': 'unet'}

    args = run_offline_training.parse_args(
        ['--project', 'p', '--max-minutes', '45', '--trainer-commit', 'abc'])
    assert args.max_minutes == 45.0 and args.trainer_commit == 'abc'
    assert args.foreground_tile_fraction is None
    tuned = run_offline_training.parse_args(
        ['--project', 'p', '--loss-type', 'combined_per_tile',
         '--foreground-tile-fraction', '0.27'])
    assert tuned.loss_type == 'combined_per_tile'
    assert tuned.foreground_tile_fraction == 0.27
    assert run_offline_training.parse_args(
        ['--project', 'p']).max_minutes is None


def test_seed_worker_makes_worker_randomness_repeatable():
    import random

    def draw():
        torch.manual_seed(123)
        run_offline_training.seed_worker(0)
        return np.random.rand(), random.random()

    first = draw()
    np.random.rand()
    assert draw() == first
    torch.manual_seed(124)
    run_offline_training.seed_worker(0)
    assert np.random.rand() != first[0]


def test_export_trainer_source_extracts_a_commit(tmp_path):
    if shutil.which('git') is None:
        pytest.skip('git is not available')
    export = run_offline_training.export_trainer_source
    try:
        src_dir, sha = export('HEAD', tmp_path)
    except subprocess.CalledProcessError:
        pytest.skip('not running from a git checkout')

    assert len(sha) == 40
    assert src_dir == tmp_path / sha[:12] / 'trainer' / 'src'
    assert (src_dir / 'trainer.py').is_file()
    # A second call reuses the existing export instead of rewriting it.
    (src_dir / 'marker').write_text('kept')
    assert export('HEAD', tmp_path) == (src_dir, sha)
    assert (src_dir / 'marker').read_text() == 'kept'


def _scripted_validation_run(root, monkeypatch, cache):
    """Drive ``Trainer.validation`` through six scripted epochs.

    The candidate is worse for three epochs (rollback), improves on the
    fourth (new checkpoint), then is worse again.
    """
    import trainer as trainer_module
    from metrics import get_metrics

    root.mkdir()
    dataset, sync_dir = _make_dataset(root)
    prepare(dataset, sync_dir, 'proj', model_type='unet')
    project = sync_dir / 'projects' / 'proj'
    args = argparse.Namespace(
        patchsize=572, maxworkers=0, maxbatchsize=2, model_type='unet',
        loss_type='auto', max_epochs_without_progress=50,
        candidate_rollback_patience=3, no_validation_cache=not cache)
    trainer = run_offline_training.build_offline_trainer(
        project, sync_dir, args)
    trainer.train_config = {
        'model_dir': str(project / 'models'),
        'dataset_dir': str(dataset / 'images'),
        'train_annot_dir': str(project / 'annotations' / 'train'),
        'val_annot_dir': str(project / 'annotations' / 'val'),
        'log_dir': str(project / 'logs'),
    }
    trainer.model = torch.nn.Linear(1, 1)
    torch.save(trainer.model.state_dict(),
               project / 'models' / '000001_1.pkl')

    run = argparse.Namespace(
        trainer=trainer, project=project, best_loads=[], rollback_loads=[],
        candidate_losses=iter([0.9, 0.9, 0.9, 0.5, 0.6, 0.6, 0.6]), trace=[])
    best_models = []

    def get_prev_model(model_dir, model_type='unet'):
        path = model_utils.get_latest_model_paths(model_dir, k=1)[0]
        model = torch.nn.Linear(1, 1)
        model.checkpoint = os.path.basename(path)[:6]
        best_models.append(model)
        run.best_loads.append(model.checkpoint)
        return model, path

    def load_model(path, model_type='unet'):
        run.rollback_loads.append(os.path.basename(path)[:6])
        return torch.nn.Linear(1, 1)

    def get_val_metrics(cnn, **_kwargs):
        if any(cnn is model for model in best_models):
            loss = 0.8 if cnn.checkpoint == '000001' else 0.5
        else:
            loss = next(run.candidate_losses)
        metrics = get_metrics(5, 1, 90, 4, 100, 1.5, loss)
        metrics.update(foreground_defined=9, foreground_file_count=2)
        return metrics

    monkeypatch.setattr(model_utils, 'get_prev_model', get_prev_model)
    monkeypatch.setattr(model_utils, 'load_model', load_model)
    monkeypatch.setattr(model_utils, 'get_val_metrics', get_val_metrics)
    monkeypatch.setattr(trainer_module, 'build_optimizer',
                        lambda model, model_type: object())

    def step():
        trainer.epoch += 1
        trainer.validation()
        assert isinstance(trainer.model, torch.nn.Module)
        run.trace.append((
            trainer.epochs_without_progress, trainer.candidate_worse_epochs,
            trainer.best_val_loss, trainer.rollbacks,
            len(os.listdir(project / 'models'))))

    run.step = step
    for _ in range(6):
        step()
    with open(project / 'logs' / 'offline_history.jsonl') as stream:
        records = [json.loads(line) for line in stream]
    run.best_losses = [r['loss'] for r in records if r['name'] == 'prev_val']
    return run


def test_best_checkpoint_validation_cache_matches_per_epoch_rescoring(
        tmp_path, monkeypatch):
    plain = _scripted_validation_run(tmp_path / 'plain', monkeypatch, False)
    cached = _scripted_validation_run(tmp_path / 'cached', monkeypatch, True)

    # Same promotions, rollback, patience counters and logged metrics...
    assert cached.trace == plain.trace
    assert cached.best_losses == plain.best_losses == [
        0.8, 0.8, 0.8, 0.8, 0.5, 0.5]
    assert cached.trainer.rollbacks == 1
    assert len(os.listdir(cached.project / 'models')) == 2

    # ...but each best checkpoint is loaded and scored once, not every epoch.
    assert plain.best_loads == ['000001'] * 4 + ['000002'] * 2
    assert cached.best_loads == ['000001', '000002']
    assert cached.trainer.validation_cache_hits == 4
    assert plain.trainer.validation_cache_hits == 0
    # The weights were still loaded for the one epoch that rolled back.
    assert cached.rollback_loads == ['000001']
    assert plain.rollback_loads == []

    # Editing a validation annotation invalidates the cached metrics.
    annotation = next((cached.project / 'annotations' / 'val').iterdir())
    stat = annotation.stat()
    os.utime(annotation, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    cached.step()
    assert cached.best_loads == ['000001', '000002', '000002']


# -- session replay ---------------------------------------------------------

def test_session_routes_split_the_remaining_scans_as_the_painter_does():
    from prepare_offline_project import (fold_routes, painter_project_order,
                                         seeded_file_order, session_routes)

    rows = _cohort()
    patient_of = {row['filename']: row['patient_id'] for row in rows}
    ordered = seeded_file_order(list(patient_of), 0)
    for fold in range(4):
        route = session_routes(rows, 4, fold, seed=0)
        # The same patients are held out as in ordinary cross-validation.
        held_out = {name for name, role in route.items() if role == 'test'}
        assert held_out == {name for name, role in
                            fold_routes(rows, 4, fold, seed=0).items()
                            if role == 'test'}
        # The rest is the painter's patient-aware split of the session order.
        session = [name for name in ordered if name not in held_out]
        train, val = painter_project_order().patient_grouped_train_val_split(
            session, patient_of)
        assert [n for n in session if route[n] == 'train'] == train
        assert [n for n in session if route[n] == 'val'] == val
        sides = {}
        for name in session:
            sides.setdefault(patient_of[name], set()).add(route[name])
        assert all(len(found) == 1 for found in sides.values())
        # Validation arrives from the first scan and keeps pace all session.
        assert route[session[0]] == 'val'
        for viewed in (12, 30, 60):
            share = sum(route[n] == 'val' for n in session[:viewed]) / viewed
            assert 0.08 < share < 0.34

    with pytest.raises(ValueError, match='between 0 and 3'):
        session_routes(rows, 4, 4)


def test_session_split_project_keeps_the_fold_held_out(tmp_path):
    dataset, sync_dir = _make_dataset(tmp_path)
    plain = prepare(dataset, sync_dir, 'cv_fold0', model_type='unet',
                    folds=3, fold=0)
    summary = prepare(dataset, sync_dir, 'replay_fold0', model_type='unet',
                      folds=3, fold=0, session_split=True)
    assert summary['session_split'] and not plain['session_split']
    load = run_offline_training.load_project
    settings = load(sync_dir / 'projects' / 'replay_fold0')[0]
    assert settings['split_method'] == 'patient_fold_session_split_v1'
    assert settings['test_file_names'] == load(
        sync_dir / 'projects' / 'cv_fold0')[0]['test_file_names']
    # Two patients remain: the first met is validation, the other training.
    assert settings['train_file_names'] and settings['val_file_names']
    assert ({name[0] for name in settings['train_file_names']}
            .isdisjoint(name[0] for name in settings['val_file_names']))

    with pytest.raises(ValueError, match='needs folds'):
        prepare(dataset, sync_dir, 'no_fold', session_split=True)


def _session_project(root, annotated):
    """Project folder with annotations: ``{stem: (role, has lesion mark)}``."""
    project = root / 'project'
    for role in ('train', 'val'):
        (project / 'annotations' / role).mkdir(parents=True)
    for stem, (role, lesion) in annotated.items():
        foreground = np.zeros(SHAPE, bool)
        foreground[5:8, 5:8] = lesion
        background = np.zeros(SHAPE, bool)
        background[40:42, 0:20] = True
        mask_io.write_annotation(
            project / 'annotations' / role / f'{stem}.png',
            foreground, background)
    return project


class _FakeTrainer:
    """Stands in for the trainer: an epoch takes ``epoch_seconds``."""

    def __init__(self, project, now, epoch_seconds=25, stop_after=None,
                 stop_reason=None):
        self.project, self.now = project, now
        self.epoch_seconds = epoch_seconds
        self.stop_after, self.final_reason = stop_after, stop_reason
        self.training, self.epoch, self.stop_reason = True, 0, None
        self.seen = []

    def train_one_epoch(self):
        self.seen.append(sorted(
            f'{path.parent.name}/{path.stem}'
            for path in (self.project / 'annotations').rglob('*.png')))
        self.now[0] += self.epoch_seconds
        if self.stop_after == len(self.seen):
            self.training = False
            self.stop_reason = self.final_reason


def _replay(tmp_path, annotated, trainer_options=None, **options):
    import run_session_replay as replay_module

    project = _session_project(tmp_path, annotated)
    now = [1000.0]
    clock = replay_module.SessionClock(timer=lambda: now[0])
    replay = replay_module.SessionReplay(
        project, [f's{index:02d}.png' for index in range(12)],
        clock=clock, **options)
    replay.stage()
    trainer = _FakeTrainer(project, now, **(trainer_options or {}))
    shown, drawn = [], []
    completed = replay.run(
        trainer, lambda label: shown.append((label, replay.revealed)),
        drawn.extend)
    return argparse.Namespace(replay=replay, trainer=trainer, shown=shown,
                              drawn=drawn, completed=completed, clock=clock,
                              project=project)


ANNOTATED = {
    's00': ('val', True), 's01': ('train', False), 's03': ('train', True),
    's05': ('train', False), 's06': ('val', False), 's09': ('train', True),
    's11': ('val', True),
}


def test_session_replay_feeds_annotations_in_viewing_order(tmp_path):
    run = _replay(tmp_path, ANNOTATED, pace_seconds=10, warmup_scans=3,
                  milestones=(4, 8, 99))
    replay, trainer = run.replay, run.trainer

    # Staging hid every annotation; training began after three scans, with
    # only what those three scans had brought.
    assert trainer.seen[0] == ['train/s01', 'val/s00']
    assert run.clock.skipped == 30
    # 25-second epochs at 10 seconds a scan: 5, 8 and 10 scans viewed at the
    # next three epoch starts, and the session ends when all 12 are viewed.
    assert trainer.seen[1] == ['train/s01', 'train/s03', 'val/s00']
    assert trainer.seen[2] == ['train/s01', 'train/s03', 'train/s05',
                               'val/s00', 'val/s06']
    assert len(trainer.seen) == replay.epochs == 4
    # The model on screen is recorded once each milestone has been reached;
    # a milestone beyond the session is dropped and the end always recorded.
    assert run.shown == [(4, 5), (8, 8), ('end', 12)]
    assert run.completed
    # Every scan after the warm-up was drawn once, in viewing order.
    assert run.drawn == [f's{index:02d}.png' for index in range(3, 12)]
    assert replay.segmented == 9
    # At the end every annotation is back where the project assigned it.
    assert sorted(f'{path.parent.name}/{path.stem}' for path in
                  (run.project / 'annotations').rglob('*.png')) == sorted(
                      f'{role}/{stem}' for stem, (role, _) in ANNOTATED.items())
    assert not list(replay.pending_dir.rglob('*.png'))
    assert replay.counts() == {
        'train_annotations': 4, 'train_with_lesion_marks': 2,
        'val_annotations': 3, 'val_with_lesion_marks': 2}


def test_session_replay_skips_idle_time_and_handles_stops(tmp_path):
    import run_session_replay as replay_module

    # One warm-up scan gives validation only: the trainer would sit idle
    # until the first training annotation, so that time is skipped.
    idle = _replay(tmp_path / 'idle', ANNOTATED, pace_seconds=10,
                   warmup_scans=1)
    assert idle.trainer.seen[0] == ['train/s01', 'val/s00']
    assert idle.clock.skipped == 20

    # Early stopping ends training, not the session: scans keep arriving
    # and later milestones show the model that was final.
    early = _replay(tmp_path / 'early', ANNOTATED, pace_seconds=10,
                    warmup_scans=3, milestones=(4, 11),
                    trainer_options=dict(stop_after=1))
    assert len(early.trainer.seen) == 1 and early.completed
    assert early.shown == [(4, 5), (11, 11), ('end', 12)]
    assert early.replay.early_stop_scan == 5

    # A STOP file abandons the replay; the rest stays staged.
    stopped = _replay(tmp_path / 'stopped', ANNOTATED, pace_seconds=10,
                      warmup_scans=3, trainer_options=dict(
                          stop_after=1, stop_reason='stop_file'))
    assert not stopped.completed and stopped.shown == []
    assert list(stopped.replay.pending_dir.rglob('*.png'))

    # A shortened session leaves later annotations staged, and a tail keeps
    # training after the last scan.
    short = _replay(tmp_path / 'short', ANNOTATED, pace_seconds=10,
                    warmup_scans=3, max_scans=6, tail_minutes=1)
    assert short.replay.revealed == 6 and short.shown == [('end', 6)]
    assert short.replay.counts()['train_annotations'] == 3
    assert {path.stem for path in short.replay.pending_dir.rglob('*.png')} == {
        's06', 's09', 's11'}
    assert short.clock.now() >= 6 * 10 + 60

    # A project can hold one replay, and every annotation must be a scan's.
    with pytest.raises(ValueError, match='already holds a replay'):
        short.replay.stage()
    stray = _session_project(tmp_path / 'stray', {'zz': ('train', False)})
    with pytest.raises(ValueError, match='belong to no scan'):
        replay_module.SessionReplay(stray, ['s00.png'], 10).stage()
    with pytest.raises(ValueError, match='positive number'):
        replay_module.SessionReplay(stray, ['s00.png'], 0)


def test_session_replay_options_and_clock():
    import run_session_replay as replay_module

    args = replay_module.parse_args(['--project', 'p'])
    assert args.pace_seconds == 20.0 and args.milestones == [25, 50, 100, 200]
    # By default the best checkpoint is re-scored every epoch, as it is live.
    assert args.no_validation_cache and not args.no_segmentation
    tuned = replay_module.parse_args(
        ['--project', 'p', '--pace-seconds', '8', '--milestones', '50, 10,10',
         '--validation-cache', '--no-rollback', '--max-scans', '80'])
    assert tuned.milestones == [10, 50] and not tuned.no_validation_cache
    assert tuned.no_rollback and tuned.max_scans == 80
    with pytest.raises(SystemExit):
        replay_module.parse_args(['--project', 'p', '--max-minutes', '5'])
    with pytest.raises(ValueError, match='from 1 up'):
        replay_module.parse_milestones('0,5')

    settings = {'file_names': ['a', 'b', 'c', 'd'], 'test_file_names': ['b']}
    assert replay_module.session_order(settings) == ['a', 'c', 'd']
    with pytest.raises(ValueError, match='no held-out scans'):
        replay_module.session_order({'file_names': ['a']})

    now = [50.0]
    clock = replay_module.SessionClock(timer=lambda: now[0])
    now[0] += 4
    clock.skip_to(30)
    assert clock.now() == 30 and clock.skipped == 26
    clock.skip_to(10)                 # never backwards
    now[0] += 5
    assert clock.now() == 35


def test_replay_scoring_pools_folds_milestone_by_milestone():
    import score_session_replay as scoring

    def milestone(label, records, viewed, minutes, shown='checkpoint'):
        return dict(milestone=label, scans_viewed=viewed,
                    session_minutes=minutes, ui_model=shown,
                    train_annotations=viewed // 2, val_annotations=1,
                    train_with_lesion_marks=1, val_with_lesion_marks=0,
                    records=records)

    early_a = [_held_out('a1', 'A', 'present', 0.99, 1, lesions=1, found=1),
               _held_out('a2', 'A', 'absent', 0.99, 1, fp_blobs=3)]
    early_b = [_held_out('b1', 'B', 'present', 0.2, 0, lesions=1),
               _held_out('b2', 'B', 'absent', 0.6, 1, fp_blobs=1)]
    late_a = [_held_out('a1', 'A', 'present', 0.9, 1, lesions=1, found=1),
              _held_out('a2', 'A', 'absent', 0.1, 0)]
    late_b = [_held_out('b1', 'B', 'present', 0.7, 1, lesions=1, found=1),
              _held_out('b2', 'B', 'absent', 0.3, 0)]
    folds = [
        [milestone(10, early_a, 10, 3.0, 'live weights'),
         milestone('end', late_a, 40, 13.0)],
        [milestone(10, early_b, 12, 4.0), milestone('end', late_b, 44, 15.0)],
    ]
    rows = scoring.pool(folds, resamples=50)
    first, last = rows
    assert [row['milestone'] for row in rows] == [10, 'end']
    assert first['positive_scans_flagged'] == 1 and first['positive_scans'] == 2
    assert first['negative_scans_cleared'] == 0
    assert first['scans_viewed'] == 11 and first['session_minutes'] == 3.5
    assert first['annotated_scans'] == 6.5
    assert first['showing'] == {'live weights': 1, 'checkpoint': 1}
    # Fold A's early model flags everything, so no cut-off clears 90%.
    assert first['positive_scans_flagged_at_target_specificity'] is None
    assert last['positive_scans_flagged'] == 2 and last['auroc'] == 1.0
    assert last['negative_scans_cleared'] == 2
    assert last['positive_scans_flagged_at_target_specificity'] == 2
    assert last['patients'] == 2 and 'auroc' in last['intervals']

    assert scoring.sensitivity_at_specificity(early_a) == (None, 1)
    # The lowest cut-off above every negative score still catches 0.7 and 0.9.
    assert scoring.sensitivity_at_specificity(late_a + late_b) == (2, 2)
    assert scoring.sensitivity_at_specificity(late_a[:1]) == (None, 1)

    # A later run is compared at the milestones both recorded.
    versus = scoring.compare_replays(rows[1:], rows)
    assert [row['milestone'] for row in versus] == ['end']
    assert versus[0]['auroc']['difference'] == 0
    with pytest.raises(ValueError, match='same milestones'):
        scoring.pool([folds[0], folds[1][:1]])


def test_cross_validation_driver_replays_a_session():
    import run_cross_validation as driver

    args = driver.parse_args([
        '--dataset', 'd', '--sync-dir', 's', '--name', 'replay',
        '--session-pace', '20', '--', '--milestones', '10,20',
        '--no-rollback'])
    assert args.session_pace == 20.0
    replay, score = driver.replay_commands(
        Path('p') / 'replay_fold1', '1', args.session_pace, args.train_args,
        Path('s'), 'replay')
    assert replay[2].endswith('run_session_replay.py')
    assert replay[-7:] == ['--gpu', '1', '--pace-seconds', '20.0',
                           '--milestones', '10,20', '--no-rollback']
    assert score[1].endswith('score_session_replay.py')
    assert score[-4:] == ['--project', str(Path('p') / 'replay_fold1'),
                          '--gpu', '1']
    assert driver.parse_args(
        ['--dataset', 'd', '--sync-dir', 's', '--name', 'cv']
    ).session_pace is None
