"""Synthetic-only checks for fresh patient-separated annotation projects."""

import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'painter/src/main/python'))
from prepare_patient_practice_project import build_patient_practice_project
from file_utils import get_annot_path, maybe_save_annotation
from instructions import fix_instruction_paths


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def _rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seal(dataset):
    hashes = {path.relative_to(dataset).as_posix(): _sha(path)
              for path in dataset.rglob('*')
              if path.is_file() and path.name != 'dataset_metadata.json'}
    (dataset / 'dataset_metadata.json').write_text(
        json.dumps({'file_sha256': hashes}), encoding='utf-8')


def _source(tmp_path, specs=None):
    # These tokens and tiny colored rectangles are entirely synthetic.
    if specs is None:
        specs = [
            ('uncertain_train', 'synthetic_u1', 'train', 'uncertain'),
            ('uncertain_val', 'synthetic_u2', 'val', 'uncertain'),
            ('train_positive_a', 'synthetic_a', 'train', 'present'),
            ('train_positive_a_repeat', 'synthetic_a', 'train', 'present'),
            ('train_positive_b', 'synthetic_b', 'train', 'present'),
            ('val_positive_c', 'synthetic_c', 'val', 'present'),
            ('val_positive_d', 'synthetic_d', 'val', 'present'),
            ('train_absent_a', 'synthetic_e', 'train', 'absent'),
            ('train_absent_b', 'synthetic_f', 'train', 'absent'),
            ('train_absent_c', 'synthetic_g', 'train', 'absent'),
            ('train_absent_d', 'synthetic_h', 'train', 'absent'),
            ('val_absent_a', 'synthetic_i', 'val', 'absent'),
            ('val_absent_b', 'synthetic_j', 'val', 'absent'),
            ('hold_uncertain', '', 'hold', 'uncertain'),
            ('hold_absent', '', 'hold', 'absent'),
        ]
    syncdir = tmp_path / 'sync'
    dataset = syncdir / 'curated_datasets' / 'synthetic_updated'
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'overlays').mkdir()
    manifest, assignments = [], []
    for rank, (stem, patient, split, decision) in enumerate(specs, 1):
        filename = stem + '.png'
        image = dataset / 'images' / filename
        Image.new('RGB', (8, 6), (rank * 11 % 256, rank * 7 % 256, 10)).save(image)
        Image.new('RGB', (8, 6), (255, rank * 5 % 256, 0)).save(
            dataset / 'overlays' / filename)
        manifest.append(dict(
            rank=rank, filename=filename, image_sha256=_sha(image),
            patient_id=patient,
            patient_identity_status=(
                'previously_mapped' if rank % 2 else 'consistent_source_manifest'
            ) if patient else 'identity_hold',
            decision=decision, prior_development_split=split if patient else '',
            reference_picture='overlays/' + filename))
        if patient:
            assignments.append(dict(filename=filename, image_sha256=_sha(image),
                                    patient_id=patient, split=split))
    _write_rows(dataset / 'manifest.csv', manifest)
    _seal(dataset)
    assignments_path = tmp_path / 'verified_assignments.csv'
    _write_rows(assignments_path, assignments)
    return dict(dataset=dataset, assignments=assignments_path, syncdir=syncdir,
                output=syncdir / 'projects' / 'synthetic_practice')


@pytest.fixture
def source(tmp_path):
    return _source(tmp_path)


def _build(source, **options):
    return build_patient_practice_project(**source, **options)


def _config(output):
    paths = list(output.glob('*.seg_proj'))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding='utf-8-sig'))


def _pure_method(relative_path, class_name, method_name):
    """Use the real path method without importing Qt or a training runtime."""
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {'os': os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


def test_fresh_project_preserves_patient_split_holds_and_uncertainty(source):
    before = _files(source['dataset'])
    mapping_before = source['assignments'].read_bytes()
    result = _build(source)
    assert isinstance(result, dict)
    output = source['output']
    config = _config(output)
    source_rows = _rows(source['dataset'] / 'manifest.csv')
    expected = {row['filename']: row for row in source_rows if row['patient_id']}
    assert len(config['file_names']) == len(set(config['file_names'])) == 13
    assert set(config['file_names']) == set(expected)
    for split in ('train', 'val'):
        assert set(config[split + '_file_names']) == {
            name for name, row in expected.items() if row['prior_development_split'] == split}
    assert not set(config['train_file_names']) & set(config['val_file_names'])
    train_ids = {expected[name]['patient_id'] for name in config['train_file_names']}
    val_ids = {expected[name]['patient_id'] for name in config['val_file_names']}
    assert not train_ids & val_ids
    assert {'uncertain_train.png', 'uncertain_val.png'} <= set(config['file_names'])

    rows = _rows(output / 'manifest.csv')
    assert len(rows) == 15
    by_name = {row['filename']: row for row in rows}
    for old in source_rows:
        current = by_name[old['filename']]
        assert current['decision'] == old['decision']
        assert current['image_sha256'] == old['image_sha256']
        assert current['split'] == (old['prior_development_split'] or 'hold')
    held = _rows(output / 'held_images.csv')
    assert {row['filename'] for row in held} == {'hold_uncertain.png', 'hold_absent.png'}
    assert all(row['split'] == 'hold' for row in held)
    patients = _rows(output / 'patient_assignments.csv')
    assert len(patients) == len({row['patient_id'] for row in patients}) == 12
    assert {row['patient_id'] for row in patients} == train_ids | val_ids
    assert all(row['split'] == ('train' if row['patient_id'] in train_ids else 'val')
               for row in patients)

    for folder in ('annotations/train', 'annotations/val', 'models', 'segmentations',
                   'messages', 'logs', 'reviews', 'review_exports'):
        assert (output / folder).is_dir()
        assert not list((output / folder).iterdir())
    assert not list(output.rglob('*.png'))
    assert config['original_model_file'] == 'random weights'
    assert config['model_type'] == 'unet'
    assert config['training_seed'] == 0
    assert config['auto_start_training_after_reviews'] == 10
    assert (source['syncdir'] / 'datasets' / config['dataset']).resolve() == (
        source['dataset'] / 'images').resolve()
    assert (source['syncdir'] / config['location']).resolve() == output.resolve()
    assert _files(source['dataset']) == before
    assert source['assignments'].read_bytes() == mapping_before
    assert (output / 'preparation_metadata.json').is_file()


def test_warmup_has_distinct_positive_patients_in_both_splits_and_stable_remainder(source):
    _build(source)
    config = _config(source['output'])
    source_rows = _rows(source['dataset'] / 'manifest.csv')
    by_name = {row['filename']: row for row in source_rows}
    warmup = [by_name[name] for name in config['file_names'][:10]]
    assert len(warmup) == 10
    for split, expected_absent in (('train', 4), ('val', 2)):
        positive = [row for row in warmup
                    if row['prior_development_split'] == split and row['decision'] == 'present']
        assert len(positive) == 2
        assert len({row['patient_id'] for row in positive}) == 2
        assert sum(row['prior_development_split'] == split and row['decision'] == 'absent'
                   for row in warmup) == expected_absent
    warmup_names = set(config['file_names'][:10])
    assert config['file_names'][10:] == [row['filename'] for row in source_rows
                                       if row['patient_id'] and row['filename'] not in warmup_names]

    second = {**source, 'output': source['output'].with_name('synthetic_repeat')}
    _build(second)
    again = _config(second['output'])
    for key in ('file_names', 'train_file_names', 'val_file_names', 'training_seed'):
        assert again[key] == config[key]


def test_small_warmup_fills_definite_before_uncertain_without_reassigning_patients(tmp_path):
    source = _source(tmp_path, [
        ('uncertain_first', 'synthetic_a', 'train', 'uncertain'),
        ('only_positive', 'synthetic_b', 'val', 'present'),
        ('only_negative', 'synthetic_c', 'train', 'absent'),
        ('uncertain_last', 'synthetic_d', 'val', 'uncertain'),
    ])
    _build(source)
    config = _config(source['output'])
    assert set(config['file_names'][:2]) == {'only_positive.png', 'only_negative.png'}
    assert config['file_names'][2:] == ['uncertain_first.png', 'uncertain_last.png']
    assert set(config['train_file_names']) == {'uncertain_first.png', 'only_negative.png'}
    assert set(config['val_file_names']) == {'only_positive.png', 'uncertain_last.png'}


class _Pixmap:
    """The file saving interface consumed by the painter; no GUI or Qt needed."""

    def __init__(self, color):
        self.image = Image.new('RGBA', (8, 6), color)

    def save(self, path, image_format):
        self.image.save(path, image_format)


def test_saved_annotations_follow_explicit_membership_even_when_counts_differ(source):
    _build(source)
    output = source['output']
    config = _config(output)
    train, val = output / 'annotations/train', output / 'annotations/val'
    chosen = [(name, train) for name in config['train_file_names'][:2]]
    chosen += [(name, val) for name in config['val_file_names'][:2]]
    for name, target in chosen:
        png_name = Path(name).stem + '.png'
        existing = get_annot_path(png_name, str(train), str(val))
        assert existing is None
        saved = maybe_save_annotation(
            str(output), _Pixmap((255, 0, 0, 255)), existing, png_name,
            str(train), str(val), fixed_target_dir=str(target))
        assert Path(saved) == target / png_name
        assert (target / png_name).is_file()
        opposite = val if target == train else train
        assert not (opposite / png_name).exists()
    # Later edits overwrite that same allocation rather than rerouting a scan.
    name, target = chosen[0]
    png_name = Path(name).stem + '.png'
    before = (target / png_name).read_bytes()
    saved = maybe_save_annotation(
        str(output), _Pixmap((0, 255, 0, 255)),
        get_annot_path(png_name, str(train), str(val)), png_name,
        str(train), str(val), fixed_target_dir=str(target))
    assert Path(saved) == target / png_name
    assert (target / png_name).read_bytes() != before


def test_outside_datasets_path_works_in_navigation_and_trainer_instructions(source):
    _build(source)
    config = _config(source['output'])
    # Follow open_project/update_file's exact path construction. A relative
    # ../curated_datasets path used to retain '..' here while NavWidget removed
    # it, making the current image impossible to find in its own navigation list.
    dataset_dir = source['syncdir'] / 'datasets' / Path(config['dataset'])
    current = os.path.join(dataset_dir, config['file_names'][0])
    get_path_list = _pure_method('painter/src/main/python/nav.py',
                                 'NavWidget', 'get_path_list')
    navigator = SimpleNamespace(all_fnames=config['file_names'])
    paths = get_path_list(navigator, os.path.dirname(current))
    assert current == os.path.abspath(current)
    assert paths.index(current) == 0
    assert all(os.path.isfile(path) for path in paths)

    outgoing = fix_instruction_paths({
        'dataset_dir': dataset_dir,
        'file_names': config['file_names'],
        'model_type': config['model_type'],
        'model_seed': config['training_seed'],
    }, source['syncdir'])
    assert not os.path.isabs(outgoing['dataset_dir'])
    fix_config_paths = _pure_method('trainer/src/trainer.py',
                                   'Trainer', 'fix_config_paths')
    restored = fix_config_paths(SimpleNamespace(sync_dir=str(source['syncdir'])), outgoing)
    assert Path(restored['dataset_dir']) == source['dataset'] / 'images'
    assert restored['file_names'] == config['file_names']
    assert restored['model_type'] == 'unet'
    assert restored['model_seed'] == 0


@pytest.mark.parametrize('invalid', [
    'missing_mapping', 'wrong_hash', 'wrong_patient', 'patient_in_both_splits',
    'prior_split_conflict', 'unknown_eligible_patient',
])
def test_invalid_verified_assignments_never_publish_project(source, invalid):
    rows = _rows(source['assignments'])
    if invalid == 'missing_mapping':
        rows = rows[1:]
    elif invalid == 'wrong_hash':
        rows[0]['image_sha256'] = '0' * 64
    elif invalid == 'wrong_patient':
        rows[0]['patient_id'] = 'synthetic_wrong_patient'
    elif invalid == 'patient_in_both_splits':
        rows.append({**rows[0], 'filename': 'outside_current_dataset.png',
                     'image_sha256': '1' * 64, 'split': 'val'})
    else:
        manifest = _rows(source['dataset'] / 'manifest.csv')
        if invalid == 'prior_split_conflict':
            manifest[0]['prior_development_split'] = 'val'
        else:
            manifest[0]['patient_id'] = 'synthetic_never_mapped'
        _write_rows(source['dataset'] / 'manifest.csv', manifest)
        _seal(source['dataset'])
    _write_rows(source['assignments'], rows)
    with pytest.raises(ValueError):
        _build(source)
    assert not source['output'].exists()


@pytest.mark.parametrize('changed', ['image', 'overlay', 'manifest', 'extra_file'])
def test_changed_frozen_bundle_is_rejected_before_project_is_published(source, changed):
    dataset = source['dataset']
    if changed == 'extra_file':
        path = dataset / 'unexpected.txt'
    elif changed == 'manifest':
        path = dataset / 'manifest.csv'
    else:
        folder = 'images' if changed == 'image' else 'overlays'
        path = dataset / folder / 'uncertain_train.png'
    path.write_bytes(b'changed synthetic fixture')
    with pytest.raises(ValueError):
        _build(source)
    assert not source['output'].exists()


def test_existing_output_is_refused_without_modification(source):
    _build(source)
    before = _files(source['output'])
    with pytest.raises((ValueError, FileExistsError)):
        _build(source)
    assert _files(source['output']) == before
