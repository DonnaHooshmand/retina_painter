"""Positive-session preparation checks using entirely synthetic records/images."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import openpyxl
from PIL import Image
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import prepare_positive_annotation_dataset as preparation
from prepare_positive_annotation_dataset import HEADERS, discover, patient_order, publish


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rows(patients=120, scans=2):
    return [dict(patient_id=f'synthetic_{p:03}', filename=f'synthetic_{p:03}_{s}.png',
                 split='val' if p % 5 == 0 else 'train')
            for p in range(patients) for s in range(scans)]


def test_patient_order_diverse_prefix_and_fixed_roles():
    rows = _rows()
    original = [dict(r) for r in rows]
    selected = patient_order(rows, 200, 0)
    assert len(selected) == 200
    assert len({r['filename'] for r in selected}) == 200
    assert len({r['patient_id'] for r in selected[:100]}) == 100
    assert len({r['patient_id'] for r in selected[:120]}) == 120
    assert Counter(r['split'] for r in selected[:10]) == {'train': 8, 'val': 2}
    assert len({r['patient_id'] for r in selected[:10]}) == 10
    assert rows == original
    by_name = {r['filename']: r for r in original}
    assert all(r['split'] == by_name[r['filename']]['split'] for r in selected)


def test_patient_order_reproducible_and_seed_changes_order():
    rows = _rows()
    assert patient_order(rows, 200, 0) == patient_order(list(reversed(rows)), 200, 0)
    assert patient_order(rows, 200, 0) != patient_order(rows, 200, 1)
    assert patient_order(rows, 100, 0) == patient_order(rows, 200, 0)[:100]


@pytest.mark.parametrize('rows,count', [(_rows(9), 10), (_rows(10, 1), 11), (_rows(), 9)])
def test_patient_order_refuses_insufficient_pool(rows, count):
    with pytest.raises(ValueError):
        patient_order(rows, count)


@pytest.fixture
def synthetic_sources(tmp_path):
    syncdir = tmp_path / 'sync'
    images = syncdir / 'datasets' / 'synthetic' / 'images'
    images.mkdir(parents=True)
    (syncdir / 'projects').mkdir()
    current = syncdir / 'curated_datasets' / 'synthetic_current'
    _csv(current / 'manifest.csv', [], ['filename', 'image_sha256'])
    workbook = tmp_path / 'synthetic_classification.xlsx'
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'RIPL Annotation'
    ws.append(HEADERS)
    assignments = []
    for i in range(20):
        # Numeric identities are synthetic and deliberately unlike real records.
        patient = str(9000000000 + i)
        filename = f'{patient}_OD_scan.png'
        image = images / filename
        Image.new('RGB', (8, 6), (i, 21, 47)).save(image)
        ws.append([f'/synthetic/{patient}/OD/scan.png', 'OD', 1, patient, 1, 0])
        if i < 10:
            assignments.append(dict(filename=filename, image_sha256=_sha(image),
                                    patient_id=patient, split='val' if i < 2 else 'train'))
    wb.save(workbook)
    assignments_path = tmp_path / 'synthetic_assignments.csv'
    _csv(assignments_path, assignments, ['filename', 'image_sha256', 'patient_id', 'split'])
    return dict(workbook=workbook, current=current, assignments=assignments_path,
                syncdir=syncdir, count=20, seed=0)


def test_discover_preserves_historical_roles_and_assigns_new_patients(synthetic_sources):
    rows, _, summary, watched, _ = discover(**synthetic_sources)
    assert summary['selected_scans'] == 20
    assert summary['patient_overlap'] == 0
    assert summary['warmup_split'] == {'train': 8, 'val': 2}
    assert summary['assignment_sources'] == {
        'preserved_verified_assignment': 10, 'new_patient_seeded_development_assignment': 10}
    assert summary['unavailable_historical_source_images'] == 0
    by_patient = {r['patient_id']: r for r in rows}
    for i in range(10):
        assert by_patient[str(9000000000 + i)]['split'] == ('val' if i < 2 else 'train')
    assert rows == discover(**synthetic_sources)[0]
    assert all(_sha(path) == digest for path, digest in watched.items())


def _publish_args(source):
    selected, exclusions, summary, watched, project_files = discover(**source)
    return dict(selected=selected, exclusions=exclusions, summary=summary, watched=watched,
                project_files=project_files,
                workbook=source['workbook'], assignments=source['assignments'],
                syncdir=source['syncdir'], output=source['syncdir'] / 'curated_datasets' / 'synthetic_session',
                project=source['syncdir'] / 'projects' / 'synthetic_session')


def test_publish_blank_project_copies_exact_images_and_preserves_sources(synthetic_sources):
    args = _publish_args(synthetic_sources)
    publish(**args)
    output, project = args['output'], args['project']
    config = json.loads((project / 'synthetic_session.seg_proj').read_text())
    assert Path(config['dataset']) == (output / 'images').resolve()
    assert config['original_model_file'] == 'random weights'
    assert config['model_type'] == 'unet'
    assert config['auto_start_training_after_reviews'] == 10
    assert config['file_names'] == [r['filename'] for r in args['selected']]
    train, val = set(config['train_file_names']), set(config['val_file_names'])
    assert not train & val
    assert train | val == set(config['file_names'])
    for folder in ('annotations/train', 'annotations/val', 'models', 'segmentations', 'reviews', 'review_exports'):
        assert (project / folder).is_dir()
        assert not list((project / folder).iterdir())
    assert len(list((output / 'images').iterdir())) == 20
    for row in args['selected']:
        assert _sha(output / 'images' / row['filename']) == row['image_sha256']
    assert all(_sha(path) == digest for path, digest in args['watched'].items())
    for root in (output, project):
        metadata = json.loads((root / 'preparation_metadata.json').read_text())
        for name, digest in metadata['file_sha256'].items():
            assert _sha(root / name) == digest


@pytest.mark.parametrize('existing', ['output', 'project'])
def test_publish_refuses_existing_output_without_changing_it(synthetic_sources, existing):
    args = _publish_args(synthetic_sources)
    args[existing].mkdir(parents=True)
    sentinel = args[existing] / 'keep.txt'
    sentinel.write_text('preserve this session')
    with pytest.raises(ValueError):
        publish(**args)
    assert sentinel.read_text() == 'preserve this session'
    assert not args['project' if existing == 'output' else 'output'].exists()


def test_publish_refuses_changed_source_before_creating_outputs(synthetic_sources):
    args = _publish_args(synthetic_sources)
    synthetic_sources['assignments'].write_text('changed source')
    with pytest.raises(ValueError):
        publish(**args)
    assert not args['output'].exists()
    assert not args['project'].exists()


def test_publish_restricts_output_roots(synthetic_sources, tmp_path):
    args = _publish_args(synthetic_sources)
    args['output'] = tmp_path / 'outside_curated_datasets'
    with pytest.raises(ValueError):
        publish(**args)
    assert not args['output'].exists()
    assert not args['project'].exists()


def test_discover_excludes_every_copy_with_conflicting_patient_identities(synthetic_sources):
    images = synthetic_sources['syncdir'] / 'datasets' / 'synthetic' / 'images'
    # Three names claim different patients but contain the very same synthetic image.
    names = [f'{9000000000 + i}_OD_scan.png' for i in (10, 11, 12)]
    source_bytes = (images / names[0]).read_bytes()
    for name in names[1:]:
        (images / name).write_bytes(source_bytes)
    rows, exclusions, summary, _, _ = discover(**dict(synthetic_sources, count=17))
    assert not set(names) & {r['filename'] for r in rows}
    assert _sha(images / names[0]) not in {r['image_sha256'] for r in rows}
    assert {r['filename'] for r in exclusions
            if r['reason'] == 'same_content_conflicting_patients'} == set(names)
    assert summary['exclusions']['same_content_conflicting_patients'] == 3
    assert summary['eligible_scans'] == 17


@pytest.mark.parametrize('seed', [-1, 2 ** 32, 0.5, '0'])
def test_discover_refuses_invalid_training_seed(synthetic_sources, seed):
    with pytest.raises(ValueError, match='Seed must be'):
        discover(**dict(synthetic_sources, seed=seed))


def test_discover_accepts_maximum_training_seed(synthetic_sources):
    _, _, summary, _, _ = discover(**dict(synthetic_sources, seed=2 ** 32 - 1))
    assert summary['seed'] == 2 ** 32 - 1


@pytest.mark.parametrize('new_file', [
    'annotations/train/synthetic_new.png',
    'reviews/review_records.json',
    'synthetic_active.seg_proj',
])
def test_publish_refuses_new_project_inputs_since_discovery(synthetic_sources, new_file):
    args = _publish_args(synthetic_sources)
    path = synthetic_sources['syncdir'] / 'projects' / 'synthetic_active' / new_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('synthetic inventory entry')
    with pytest.raises(ValueError, match='inventory changed before'):
        publish(**args)
    assert not args['output'].exists()
    assert not args['project'].exists()


def test_discover_refuses_project_inventory_change_while_selecting(synthetic_sources, monkeypatch):
    original_order = preparation.patient_order

    def order_then_add_project(*args, **kwargs):
        rows = original_order(*args, **kwargs)
        path = synthetic_sources['syncdir'] / 'projects' / 'synthetic_new' / 'synthetic_new.seg_proj'
        path.parent.mkdir()
        path.write_text('{}')
        return rows

    monkeypatch.setattr(preparation, 'patient_order', order_then_add_project)
    with pytest.raises(ValueError, match='inventory changed during candidate selection'):
        discover(**synthetic_sources)


def test_publish_refuses_inventory_change_while_copying(synthetic_sources, monkeypatch):
    args = _publish_args(synthetic_sources)
    original_copy = preparation.shutil.copy2
    path = synthetic_sources['syncdir'] / 'projects' / 'synthetic_active' / 'annotations/train/new.png'

    def copy_then_add_annotation(*copy_args, **copy_kwargs):
        result = original_copy(*copy_args, **copy_kwargs)
        if not path.exists():
            path.parent.mkdir(parents=True)
            path.write_bytes(b'synthetic changed inventory')
        return result

    monkeypatch.setattr(preparation.shutil, 'copy2', copy_then_add_annotation)
    with pytest.raises(ValueError, match='inventory changed during publication'):
        publish(**args)
    assert not args['output'].exists()
    assert not args['project'].exists()
