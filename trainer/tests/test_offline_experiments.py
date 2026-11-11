"""Tests for the offline experiment tools in ``trainer/experiments``."""

import argparse
import csv
import json
import math
import os
import sys

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
