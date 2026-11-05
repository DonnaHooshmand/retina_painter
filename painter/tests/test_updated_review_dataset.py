"""Synthetic-only checks for updating references without creating a project."""

import csv
import hashlib
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
from prepare_updated_review_dataset import build_updated_dataset
from review_utils import export_completed_reviews, load_review, save_review


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _files(root):
    return {path.relative_to(root): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def _project(root, syncdir, name, specs):
    """Create small colored rectangles; every identity is a synthetic token."""
    project = root / name
    project.mkdir()
    images = syncdir / 'datasets' / name
    images.mkdir(parents=True)
    manifest = images / 'manifest.csv'
    rows = []
    for rank, spec in enumerate(specs, 1):
        filename, color, decision, patient = spec[:4]
        empty_positive = len(spec) > 4 and spec[4]
        image = images / filename
        Image.new('RGB', (6, 5), (color, 10, 30)).save(image)
        # Retain a model region for explicit absent reviews, verifying that the
        # decision overrides it instead of exporting a residual positive mask.
        prediction = np.zeros((5, 6, 4), dtype=np.uint8)
        if not empty_positive:
            prediction[3, 3] = (0, 255, 255, 180)
        segmentation = project / (Path(filename).stem + '_prediction.png')
        Image.fromarray(prediction).save(segmentation)
        correction = np.zeros((5, 6, 4), dtype=np.uint8)
        correction[2, 2] = (0, 255, 0, 180)
        if decision != 'absent' and not empty_positive:
            correction[1, 1] = (255, 0, 0, 180)
        annotation = project / (Path(filename).stem + '_correction.png')
        Image.fromarray(correction).save(annotation)
        save_review(project, filename, decision, True, segmentation,
                    annotation, use_prediction=True, image_path=image)
        rows.append(dict(rank=rank, filename=filename, patient_id=patient,
                         patient_identity_status='verified' if patient else 'unmapped',
                         sha256=_sha256(image)))
    names = [spec[0] for spec in specs]
    (project / (name + '.seg_proj')).write_text(json.dumps(dict(
        dataset=name, location='projects/' + name, file_names=names,
        train_file_names=names[:1], val_file_names=names[1:], model_type='unet')),
        encoding='utf-8')
    _write_rows(manifest, rows)
    return project, images, manifest


def _freeze(project, manifest, destination):
    destination.mkdir()
    snapshot = destination / 'source_snapshot'
    snapshot.mkdir()
    config = next(project.glob('*.seg_proj'))
    shutil.copyfile(config, snapshot / config.name)
    shutil.copytree(project / 'reviews', snapshot / 'reviews')
    shutil.copyfile(manifest, snapshot / 'dataset_manifest.csv')
    order = json.loads(config.read_text(encoding='utf-8'))['file_names']
    export_completed_reviews(project, order, destination / 'completed_reviews')
    _refresh_freeze_hashes(destination)


def _refresh_freeze_hashes(destination):
    files = {path.relative_to(destination).as_posix(): _sha256(path)
             for path in destination.rglob('*')
             if path.is_file() and path.name != 'freeze_metadata.json'}
    (destination / 'freeze_metadata.json').write_text(json.dumps(dict(
        file_sha256=files, completed_review_count=len(_rows(
            destination / 'completed_reviews/reviews.csv')))), encoding='utf-8')


def _source(tmp_path, original_specs=None, followup_specs=None):
    syncdir = tmp_path / 'sync'
    if original_specs is None:
        original_specs = [
            ('repeat.png', 20, 'present', 'synthetic_patient_a'),
            ('retained.png', 60, 'absent', 'synthetic_patient_b'),
            ('uncertain.png', 100, 'uncertain', 'synthetic_patient_c'),
        ]
    if followup_specs is None:
        followup_specs = [
            ('repeat-renamed.png', 20, 'absent', 'synthetic_patient_a'),
            ('new.png', 140, 'present', ''),
        ]
    original, original_images, manifest = _project(
        tmp_path, syncdir, 'original', original_specs)
    followup, followup_images, followup_manifest = _project(
        tmp_path, syncdir, 'followup', followup_specs)
    frozen = tmp_path / 'frozen_followup'
    _freeze(followup, followup_manifest, frozen)
    return dict(original=original, frozen=frozen, syncdir=syncdir,
                output=tmp_path / 'updated', manifest=manifest,
                original_images=original_images, followup_images=followup_images)


@pytest.fixture
def updated_source(tmp_path):
    return _source(tmp_path)


def _build(source):
    return build_updated_dataset(
        source['original'], source['frozen'], source['syncdir'],
        source['output'], original_manifest=source['manifest'])


def test_latest_read_wins_for_renamed_repeat_and_both_reads_survive(updated_source):
    source = updated_source
    before = {root: _files(root) for root in (
        source['original'], source['frozen'],
        source['original_images'], source['followup_images'])}

    result = _build(source)

    assert isinstance(result, dict)
    output = source['output']
    rows = _rows(output / 'manifest.csv')
    by_name = {row['filename']: row for row in rows}
    assert set(by_name) == {'repeat.png', 'retained.png', 'uncertain.png', 'new.png'}
    assert [int(row['rank']) for row in rows] == [1, 2, 3, 4]
    repeat = by_name['repeat.png']
    assert repeat['decision'] == 'absent'
    assert repeat['prior_decision'] == 'present'
    assert repeat['decision_changed'] == 'True'
    assert repeat['current_round'] == 'followup'
    assert repeat['image_sha256'] == _sha256(source['original_images'] / 'repeat.png')
    assert not np.asarray(Image.open(output / repeat['binary_mask'])).any()
    assert not np.asarray(Image.open(output / 'doctor_foreground_masks/repeat.png')).any()
    assert (output / 'images/repeat.png').read_bytes() == (
        source['original_images'] / 'repeat.png').read_bytes()
    assert not (output / 'images/repeat-renamed.png').exists()
    assert by_name['retained.png']['current_round'] == 'original'
    assert by_name['new.png']['current_round'] == 'followup'
    assert by_name['new.png']['patient_id'] == ''

    history = _rows(output / 'review_history.csv')
    assert len(history) == 5
    repeated_reads = [row for row in history if row['filename'] == 'repeat.png']
    assert {(row['round'], row['source_filename'], row['decision'],
             row['selected_current_reference']) for row in repeated_reads} == {
        ('original', 'repeat.png', 'present', 'False'),
        ('followup', 'repeat-renamed.png', 'absent', 'True'),
    }
    assert sum(row['selected_current_reference'] == 'True' for row in history) == 4
    for side, original in (('original', source['original']),
                           ('followup', source['frozen'] / 'source_snapshot')):
        preserved = _files(output / 'provenance' / side)
        assert preserved
        for path in (original / 'reviews').rglob('*'):
            if path.is_file():
                assert path.read_bytes() in preserved.values()
    for root, contents in before.items():
        assert _files(root) == contents


def test_every_scan_gets_reference_without_project_or_new_split(updated_source):
    _build(updated_source)
    output = updated_source['output']
    rows = _rows(output / 'manifest.csv')
    for row in rows:
        assert (output / 'images' / row['filename']).is_file()
        assert (output / row['reference_picture']).is_file()
        assert 'split' not in row
    for folder in ('practice_project', 'annotations', 'models', 'segmentations',
                   'train', 'val', 'test'):
        assert not (output / folder).exists()
    assert not list(output.glob('*.seg_proj'))
    # Project configurations retained under provenance are evidence only.
    assert all('provenance' in path.relative_to(output).parts
               for path in output.rglob('*.seg_proj'))


def test_uncertain_reference_is_preserved_without_a_definitive_mask(updated_source):
    _build(updated_source)
    output = updated_source['output']
    row = next(row for row in _rows(output / 'manifest.csv')
               if row['filename'] == 'uncertain.png')
    assert row['decision'] == 'uncertain'
    assert row['binary_mask'] == '' and row['ground_truth_mask'] == ''
    assert row['reference_status'] == 'uncertain_not_definitive_reference'
    assert not (output / 'ground_truth_binary/uncertain.png').exists()
    assert not (output / 'ground_truth/uncertain.png').exists()
    assert np.asarray(Image.open(output / 'doctor_foreground_masks/uncertain.png')).any()


@pytest.mark.parametrize('decision,empty_positive', [('uncertain', False), ('present', True)])
def test_latest_nondefinitive_read_withholds_earlier_positive_mask(
        tmp_path, decision, empty_positive):
    source = _source(tmp_path, followup_specs=[
        ('repeat-renamed.png', 20, decision, 'synthetic_patient_a', empty_positive)])
    _build(source)
    output = source['output']
    row = next(row for row in _rows(output / 'manifest.csv') if row['filename'] == 'repeat.png')
    assert row['decision'] == decision
    assert row['prior_decision'] == 'present'
    assert row['current_round'] == 'followup'
    assert row['binary_mask'] == '' and row['ground_truth_mask'] == ''
    assert not (output / 'ground_truth_binary/repeat.png').exists()
    assert (output / row['reference_picture']).is_file()
    expected = ('positive_review_has_empty_mask_needs_correction' if empty_positive
                else 'uncertain_not_definitive_reference')
    assert row['reference_status'] == expected


def test_identity_conflict_is_recorded_without_inventing_a_patient_or_split(tmp_path):
    source = _source(tmp_path, followup_specs=[
        ('repeat-renamed.png', 20, 'absent', 'synthetic_patient_different')])
    _build(source)
    row = next(row for row in _rows(source['output'] / 'manifest.csv')
               if row['filename'] == 'repeat.png')
    assert row['patient_id'] == ''
    assert row['patient_identity_status'] == 'conflict_hold'
    assert 'split' not in row
    assert (source['output'] / row['reference_picture']).is_file()
    history = [row for row in _rows(source['output'] / 'review_history.csv')
               if row['filename'] == 'repeat.png']
    assert {row['patient_id'] for row in history} == {
        'synthetic_patient_a', 'synthetic_patient_different'}


@pytest.mark.parametrize('corrupt', [
    'original_snapshot', 'original_image', 'followup_image', 'frozen_snapshot',
    'frozen_records', 'frozen_mask', 'frozen_manifest',
])
def test_source_tampering_never_publishes_a_dataset(updated_source, corrupt):
    source = updated_source
    snapshot = source['frozen'] / 'source_snapshot'
    if corrupt == 'original_snapshot':
        path = source['original'] / load_review(
            source['original'], 'repeat.png')['annotation_snapshot']
    elif corrupt == 'original_image':
        path = source['original_images'] / 'repeat.png'
    elif corrupt == 'followup_image':
        path = source['followup_images'] / 'repeat-renamed.png'
    elif corrupt == 'frozen_snapshot':
        path = snapshot / load_review(snapshot, 'repeat-renamed.png')['segmentation_snapshot']
    elif corrupt == 'frozen_records':
        path = snapshot / 'reviews/review_records.json'
    elif corrupt == 'frozen_mask':
        path = source['frozen'] / 'completed_reviews/binary_masks/repeat-renamed.png'
    else:
        path = snapshot / 'dataset_manifest.csv'
    path.write_bytes(b'changed synthetic fixture')
    with pytest.raises(ValueError):
        _build(source)
    assert not source['output'].exists()


def test_unlisted_frozen_file_is_rejected(updated_source):
    (updated_source['frozen'] / 'unexpected.txt').write_text('synthetic extra file')
    with pytest.raises(ValueError):
        _build(updated_source)
    assert not updated_source['output'].exists()


@pytest.mark.parametrize('corrupt', ['decision', 'mask'])
def test_frozen_export_must_agree_with_preserved_review(updated_source, corrupt):
    frozen = updated_source['frozen']
    csv_path = frozen / 'completed_reviews/reviews.csv'
    rows = _rows(csv_path)
    row = next(row for row in rows if row['filename'] == 'repeat-renamed.png')
    if corrupt == 'decision':
        row['decision'] = 'present'
    else:
        mask_path = frozen / 'completed_reviews' / row['binary_mask']
        Image.new('L', (6, 5), 255).save(mask_path)
        row['binary_mask_sha256'] = _sha256(mask_path)
        row['foreground_pixels'] = '30'
    _write_rows(csv_path, rows)
    # An internally inconsistent freeze cannot become valid merely by hashing
    # its files again: the export must match the underlying completed review.
    _refresh_freeze_hashes(frozen)
    with pytest.raises(ValueError):
        _build(updated_source)
    assert not updated_source['output'].exists()


@pytest.mark.parametrize('which_round', ['original', 'followup'])
def test_duplicate_image_content_within_a_round_is_rejected(tmp_path, which_round):
    specs = [
        ('duplicate-one.png', 180, 'absent', 'synthetic_patient_a'),
        ('duplicate-two.png', 180, 'absent', 'synthetic_patient_a'),
    ]
    source = _source(tmp_path, **{which_round + '_specs': specs})
    with pytest.raises(ValueError):
        _build(source)
    assert not source['output'].exists()


@pytest.mark.parametrize('filename', ['repeat.png', 'retained.jpg'])
def test_same_name_or_mask_stem_for_different_images_is_rejected(tmp_path, filename):
    source = _source(tmp_path, followup_specs=[
        (filename, 200, 'absent', 'synthetic_patient_a')])
    with pytest.raises(ValueError):
        _build(source)
    assert not source['output'].exists()


def test_existing_output_is_refused_without_modification(updated_source):
    _build(updated_source)
    before = _files(updated_source['output'])
    with pytest.raises((ValueError, FileExistsError)):
        _build(updated_source)
    assert _files(updated_source['output']) == before
