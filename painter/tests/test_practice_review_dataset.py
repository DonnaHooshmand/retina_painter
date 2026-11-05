"""Synthetic-only checks for fresh practice projects and separate references."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'painter/src/main/python'))
from prepare_practice_review_dataset import build_practice_dataset
from review_utils import load_review, save_review


_fixture_spec = importlib.util.spec_from_file_location(
    '_practice_curated_fixtures', Path(__file__).with_name('test_curated_training_dataset.py'))
_fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixtures)


def _rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _files(root):
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture
def practice_source(tmp_path):
    project, original_images, manifest = _fixtures.source.__wrapped__(tmp_path)
    syncdir = tmp_path / 'sync'
    images = syncdir / 'datasets/source'
    images.parent.mkdir(parents=True)
    shutil.copytree(original_images, images)
    config_path = project / 'source.seg_proj'
    config = json.loads(config_path.read_text())
    config.update(dataset='source', location='projects/source',
                  training_seed=17, image_order_seed=17, model_type='unet')
    config_path.write_text(json.dumps(config))
    replay = tmp_path / 'replay'
    _fixtures.prepare(project, images, manifest, replay)
    output = syncdir / 'datasets/curated/_datasets/practice'
    return project, replay, syncdir, output


def _build(source):
    project, replay, syncdir, output = source
    return build_practice_dataset(project, replay, syncdir, output)


def test_fresh_project_preserves_patient_routing_without_preloading_labels(practice_source):
    project, replay, syncdir, output = practice_source
    before = {root: _files(root) for root in (project, replay, syncdir / 'datasets/source')}

    result = _build(practice_source)

    assert isinstance(result, dict)
    config_path = output / 'practice_project' / (output.name + '.seg_proj')
    config = json.loads(config_path.read_text())
    expected = _rows(replay / 'manifest.csv')
    assert set(config['file_names']) == {row['filename'] for row in expected}
    assert set(config['train_file_names']) == {
        row['filename'] for row in expected if row['split'] == 'train'}
    assert set(config['val_file_names']) == {
        row['filename'] for row in expected if row['split'] == 'val'}
    assert set(config['train_file_names']).isdisjoint(config['val_file_names'])
    assert (syncdir / 'datasets' / config['dataset']).resolve() == (output / 'images').resolve()
    assert config['model_type'] == 'unet'
    practice = config_path.parent
    for relative in ('annotations/train', 'annotations/val', 'models', 'segmentations',
                     'messages', 'logs', 'reviews', 'review_exports'):
        directory = practice / relative
        assert directory.is_dir()
        assert not list(directory.iterdir())
    assert not list(practice.rglob('review_records.json'))
    for root, contents in before.items():
        assert _files(root) == contents
    for row in expected:
        assert (output / 'images' / row['filename']).read_bytes() == (
            syncdir / 'datasets/source' / row['filename']).read_bytes()


def test_uncertainty_is_retained_and_absent_masks_are_truly_blank(practice_source):
    _, _, _, output = practice_source
    _build(practice_source)
    rows = _rows(output / 'ground_truth_status.csv')
    assert len(rows) == 5
    assert {row['decision'] for row in rows} == {'present', 'absent', 'uncertain'}
    for directory in ('doctor_foreground_reference_overlays', 'ground_truth_audit_overlays',
                      'doctor_foreground_overlays', 'doctor_foreground_masks'):
        assert {path.name for path in (output / directory).glob('*.png')} == {
            row['filename'] for row in rows}
    binary = output / 'ground_truth_binary'
    assert len(list(binary.glob('*.png'))) == 4
    for row in rows:
        mask_path = binary / row['filename']
        if row['decision'] == 'uncertain':
            assert not mask_path.exists()
            assert row['binary_mask'] == ''
            assert row['ground_truth_mask'] == ''
        else:
            assert mask_path.is_file()
            mask = np.asarray(Image.open(mask_path))
            assert set(np.unique(mask)).issubset({0, 255})
            assert bool(mask.any()) == (row['decision'] == 'present')
    # Sparse foreground remains available as a reference even for uncertainty.
    uncertain = next(row for row in rows if row['decision'] == 'uncertain')
    assert np.asarray(Image.open(output / 'doctor_foreground_masks' / uncertain['filename'])).any()


def test_unmapped_completed_review_is_exported_but_not_in_training_project(practice_source):
    project, replay, syncdir, output = practice_source
    name = 'unmapped.png'
    Image.new('RGB', (6, 5), (50, 40, 30)).save(syncdir / 'datasets/source' / name)
    segmentation = project / 'unmapped_segmentation.png'
    Image.new('RGBA', (6, 5)).save(segmentation)
    save_review(project, name, 'absent', True, segmentation, use_prediction=False)
    config_path = project / 'source.seg_proj'
    config = json.loads(config_path.read_text())
    config['file_names'].append(name)
    config_path.write_text(json.dumps(config))
    # The frozen cohort maps five files; the source also has this completed review.
    metadata_path = replay / 'replay_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(source_reviews_sha256=_fixtures.sha256(project / 'reviews/review_records.json'),
                    source_project_sha256=_fixtures.sha256(config_path))
    metadata_path.write_text(json.dumps(metadata))

    _build(practice_source)

    row = next(row for row in _rows(output / 'ground_truth_status.csv')
               if row['filename'] == name)
    assert row['split'] == 'hold'
    assert row['patient_id'] == ''
    for directory in ('images', 'doctor_foreground_reference_overlays',
                      'ground_truth_audit_overlays', 'doctor_foreground_masks', 'ground_truth_binary'):
        assert (output / directory / name).is_file()
    fresh = json.loads((output / 'practice_project' / (output.name + '.seg_proj')).read_text())
    for key in ('file_names', 'train_file_names', 'val_file_names'):
        assert name not in fresh[key]
    assert len(fresh['file_names']) == 5


def test_existing_destination_is_refused_without_changes(practice_source):
    output = practice_source[3]
    _build(practice_source)
    before = _files(output)
    with pytest.raises((ValueError, FileExistsError)):
        _build(practice_source)
    assert _files(output) == before


def test_positive_empty_reference_is_not_exported_as_a_negative_mask(practice_source):
    project, replay, syncdir, output = practice_source
    save_review(project, 'scan0.png', 'present', True, project / 'scan0.png',
                annotation_path=None, use_prediction=True)
    updated_replay = replay.parent / 'replay_empty_positive'
    _fixtures.prepare(project, syncdir / 'datasets/source',
                      replay.parent / 'source.csv', updated_replay)

    build_practice_dataset(project, updated_replay, syncdir, output)

    row = next(row for row in _rows(output / 'ground_truth_status.csv')
               if row['filename'] == 'scan0.png')
    assert row['decision'] == 'present'
    assert row['binary_mask'] == ''
    assert row['ground_truth_mask'] == ''
    assert not (output / 'ground_truth_binary/scan0.png').exists()
    assert (output / 'ground_truth_audit_overlays/scan0.png').exists()


@pytest.mark.parametrize('corrupt', ['snapshot', 'review_records', 'source_image'])
def test_bad_source_hash_does_not_publish(practice_source, corrupt):
    project, _, syncdir, output = practice_source
    if corrupt == 'snapshot':
        path = project / load_review(project, 'scan0.png')['annotation_snapshot']
    elif corrupt == 'review_records':
        path = project / 'reviews/review_records.json'
    else:
        path = syncdir / 'datasets/source/scan0.png'
    path.write_bytes(b'corrupted synthetic fixture')
    with pytest.raises(ValueError):
        _build(practice_source)
    assert not output.exists()
