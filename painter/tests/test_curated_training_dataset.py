"""Synthetic-only checks for typed references and frozen patient routing."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/main/python'))
from prepare_curated_training_dataset import (
    prepare_curated_training_dataset, read_manifest, sha256, write_csv)
import prepare_curated_training_dataset as exporter
from review_utils import load_review, save_review


def prepare(project, images, manifest, replay):
    """Build a small synthetic frozen replay, independent of older prep tools."""
    replay.mkdir()
    shutil.copytree(images, replay / 'images')
    rows, assignments = [], {}
    for row in read_manifest(manifest, ('filename', 'patient_id')):
        name, patient = row['filename'], row['patient_id']
        split = 'val' if patient in ('001', '003') else 'train'
        assignments[patient] = split
        record = load_review(project, name)
        eligible = record['decision'] != 'uncertain' and bool(record.get('annotation_snapshot'))
        annotation_hash, red, green = '', 0, 0
        if eligible:
            path = project / record['annotation_snapshot']
            rgba = np.asarray(Image.open(path).convert('RGBA'))
            red = int(((rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)).sum())
            green = int(((rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)).sum())
            destination = replay / 'annotations' / split / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            annotation_hash = sha256(path)
        rows.append(dict(filename=name, patient_id=patient, split=split,
                         decision=record['decision'], image_sha256=sha256(images / name),
                         annotation_sha256=annotation_hash, training_eligible=eligible,
                         red_pixels=red, green_pixels=green))
    write_csv(replay / 'manifest.csv', rows, list(rows[0]))
    write_csv(replay / 'patient_assignments.csv',
              [dict(patient_id=patient, split=split) for patient, split in assignments.items()],
              ['patient_id', 'split'])
    (replay / 'replay_metadata.json').write_text(json.dumps(dict(
        final_test_created=False, manifest_sha256=sha256(replay / 'manifest.csv'),
        source_reviews_sha256=sha256(project / 'reviews/review_records.json'),
        source_project_sha256=sha256(project / 'source.seg_proj'))))
    (replay / 'README.txt').write_text('Original replay documentation must remain unchanged.')


@pytest.fixture
def source(tmp_path):
    project, images = tmp_path / 'source_project', tmp_path / 'source_images'
    project.mkdir()
    images.mkdir()
    rows, names = [], []
    for index, decision in enumerate(['present', 'present', 'absent', 'absent', 'uncertain']):
        name = f'scan{index}.png'
        names.append(name)
        Image.new('RGB', (6, 5), (index * 20, 0, 0)).save(images / name)
        segmentation = project / name
        Image.new('RGBA', (6, 5)).save(segmentation)
        annotation = project / f'annotation{index}.png'
        pixels = np.zeros((5, 6, 4), dtype=np.uint8)
        pixels[2, 2] = [0, 255, 0, 180]
        if decision != 'absent':
            pixels[1, 1] = [255, 0, 0, 180]
        Image.fromarray(pixels).save(annotation)
        save_review(project, name, decision, True, segmentation, annotation, use_prediction=False)
        rows.append(dict(filename=name, patient_id=f'00{index if index < 4 else 0}',
                         sha256=sha256(images / name)))
    (project / 'source.seg_proj').write_text(json.dumps(dict(file_names=names)))
    manifest = tmp_path / 'source.csv'
    write_csv(manifest, rows, list(rows[0]))
    return project, images, manifest


def test_preserves_patient_routes_and_distinguishes_reference_types(source, tmp_path):
    project, images, manifest = source
    replay = tmp_path / 'replay'
    prepare(project, images, manifest, replay)
    output = tmp_path / 'curated'
    before = {p.relative_to(replay): p.read_bytes() for p in replay.rglob('*') if p.is_file()}
    result = prepare_curated_training_dataset(replay, output, project)
    assert result['images'] == 5 and result['reviewed_masks'] == 4
    assert not result['final_test_created']
    assert all((output / name).read_bytes() == data for name, data in before.items())
    rows = read_manifest(output / 'ground_truth/bscan_labels.csv', ('filename', 'decision'))
    uncertain = next(row for row in rows if row['decision'] == 'uncertain')
    assert uncertain['binary_label'] == '' and uncertain['binary_evaluation_eligible'] == 'False'
    index = read_manifest(output / 'ground_truth/sparse_mask_index.csv', ('filename', 'decision'))
    marked = next(row for row in index if row['decision'] == 'present')
    red = np.asarray(Image.open(output / 'ground_truth' / marked['foreground_mask']))
    known = np.asarray(Image.open(output / 'ground_truth' / marked['known_pixel_mask']))
    assert (red > 0).sum() == 1 and (known > 0).sum() == 2
    assert not known[0, 0]  # Untouched pixels were not converted into background.
    assert next(row for row in index if row['decision'] == 'uncertain')['foreground_mask'] == ''
    metadata = json.loads((output / 'curated_dataset_metadata.json').read_text())
    assert all(sha256(output / name) == digest for name, digest in metadata['file_sha256'].items())
    with pytest.raises(ValueError):
        prepare_curated_training_dataset(replay, output, project)


def test_positive_empty_mask_stays_positive_and_is_withheld(source, tmp_path):
    project, images, manifest = source
    save_review(project, 'scan0.png', 'present', True, project / 'scan0.png',
                annotation_path=None, use_prediction=True)
    replay = tmp_path / 'replay'
    prepare(project, images, manifest, replay)
    output = tmp_path / 'curated'
    result = prepare_curated_training_dataset(replay, output, project)
    assert result['reviewed_masks'] == 3
    labels = read_manifest(output / 'ground_truth/bscan_labels.csv', ('filename', 'decision'))
    label = next(row for row in labels if row['filename'] == 'scan0.png')
    assert label['decision'] == 'present' and label['binary_label'] == '1'
    rows = read_manifest(output / 'ground_truth/reviewed_mask_index.csv', ('filename', 'status'))
    row = next(row for row in rows if row['filename'] == 'scan0.png')
    assert not row['mask'] and row['status'] == 'positive_review_has_empty_mask_needs_correction'


@pytest.mark.parametrize('corrupt', ['records', 'snapshot'])
def test_source_changes_do_not_publish_ground_truth(source, tmp_path, corrupt):
    project, images, manifest = source
    replay = tmp_path / 'replay'
    prepare(project, images, manifest, replay)
    path = (project / 'reviews/review_records.json' if corrupt == 'records'
            else project / load_review(project, 'scan0.png')['annotation_snapshot'])
    path.write_bytes(b'changed')
    output = tmp_path / 'curated'
    with pytest.raises(ValueError):
        prepare_curated_training_dataset(replay, output, project)
    assert not output.exists()


def test_cli_failure_does_not_print_sensitive_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['prepare', '--replay', str(tmp_path / 'private_patient_example'),
                                   '--output', str(tmp_path / 'output')])
    with pytest.raises(SystemExit):
        exporter.main()
    captured = capsys.readouterr()
    assert 'private_patient_example' not in captured.out + captured.err
