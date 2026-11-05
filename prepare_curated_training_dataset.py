"""Publish a local, patient-separated development dataset with typed references.

No downloads, model training, changes to source projects, or patient-level stdout.
An existing output is never replaced. Failed staging folders remain unpublished.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError('Manifest is missing required columns')
        return list(reader)


def _fingerprints(root):
    if not root.is_dir():
        raise ValueError('Source directory is missing')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Source links are not supported')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def _validate_replay(root, fingerprints):
    metadata = json.loads((root / 'replay_metadata.json').read_text(encoding='utf-8-sig'))
    if metadata.get('final_test_created') is not False:
        raise ValueError('Expected an explicitly designated development replay')
    if fingerprints.get('manifest.csv') != metadata.get('manifest_sha256'):
        raise ValueError('Frozen manifest hash mismatch')
    rows = read_manifest(root / 'manifest.csv', (
        'filename', 'patient_id', 'split', 'decision', 'image_sha256',
        'annotation_sha256', 'training_eligible', 'red_pixels', 'green_pixels'))
    if not rows:
        raise ValueError('Replay is empty')
    assignments = read_manifest(root / 'patient_assignments.csv', ('patient_id', 'split'))
    patients = {}
    for row in assignments:
        if not row['patient_id'] or row['patient_id'] in patients or row['split'] not in ('train', 'val'):
            raise ValueError('Invalid or duplicated patient assignment')
        patients[row['patient_id']] = row['split']
    names, stems, content_roles, sizes = set(), set(), {}, {}
    expected_images, expected_annotations = set(), set()
    for row in rows:
        name, split = row['filename'], row['split']
        if (not name or '/' in name or '\\' in name or ':' in name
                or name in ('.', '..') or name.casefold() in names
                or Path(name).stem.casefold() in stems):
            raise ValueError('Invalid or ambiguous image filename')
        names.add(name.casefold())
        stems.add(Path(name).stem.casefold())
        if split not in ('train', 'val') or patients.get(row['patient_id']) != split:
            raise ValueError('Patient routing mismatch')
        if row['decision'] not in ('present', 'absent', 'uncertain', 'unreviewed'):
            raise ValueError('Unknown scan decision')
        if row['training_eligible'] not in ('True', 'False'):
            raise ValueError('Invalid supervision eligibility')
        image_key = 'images/' + name
        expected_images.add(image_key)
        digest = fingerprints.get(image_key)
        if not digest or digest != row['image_sha256']:
            raise ValueError('Image hash mismatch')
        if digest in content_roles and content_roles[digest] != split:
            raise ValueError('Image content overlaps patient partitions')
        content_roles[digest] = split
        with Image.open(root / image_key) as image:
            image.load()
            sizes[name] = image.size
        annotation_key = 'annotations/' + split + '/' + Path(name).stem + '.png'
        eligible = row['training_eligible'] == 'True'
        if eligible:
            expected_annotations.add(annotation_key)
            if row['decision'] == 'uncertain':
                raise ValueError('Uncertain corrections must be withheld')
            if fingerprints.get(annotation_key) != row['annotation_sha256'] or not row['annotation_sha256']:
                raise ValueError('Sparse annotation hash mismatch')
            with Image.open(root / annotation_key) as image:
                if image.mode != 'RGBA' or image.size != sizes[name]:
                    raise ValueError('Sparse annotation shape or mode mismatch')
                rgba = np.asarray(image)
            red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
            green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
            if (not (red | green).any() or (red & green).any()
                    or int(red.sum()) != int(row['red_pixels'])
                    or int(green.sum()) != int(row['green_pixels'])
                    or (row['decision'] == 'absent' and red.any())):
                raise ValueError('Sparse correction semantics mismatch')
    actual_images = {key for key in fingerprints if key.startswith('images/')}
    actual_annotations = {key for key in fingerprints if key.startswith('annotations/')}
    if actual_images != expected_images or actual_annotations != expected_annotations:
        raise ValueError('Unexpected or missing image or annotation files')
    return rows, patients, sizes


def _png(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path, format='PNG')


def _snapshot(project, record, kind, watched, expected_size):
    relative = record.get(kind + '_snapshot')
    if not relative:
        if kind == 'annotation':
            return None
        raise ValueError('Required completed-review snapshot is missing')
    path = project / relative
    if (Path(relative).is_absolute() or not path.resolve().is_relative_to(project)
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Review snapshot path is not confined to the source project')
    digest = sha256(path)
    if digest != record.get(kind + '_sha256'):
        raise ValueError('Completed-review snapshot hash mismatch')
    with Image.open(path) as image:
        image.load()
        if image.size != expected_size:
            raise ValueError('Completed-review snapshot dimensions mismatch')
    watched[path] = digest
    return path


def _review_source(replay, project):
    if project is None:
        return {}, {}, {}
    project = Path(project).resolve()
    metadata_path = replay / 'provenance/base_replay/replay_metadata.json'
    if not metadata_path.exists():
        metadata_path = replay / 'replay_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    configs = list(project.glob('*.seg_proj'))
    if len(configs) != 1:
        raise ValueError('Expected one source review project configuration')
    records_path = project / 'reviews/review_records.json'
    raw = records_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    config_hash = sha256(configs[0])
    if (digest != metadata.get('source_reviews_sha256')
            or config_hash != metadata.get('source_project_sha256')):
        raise ValueError('Review source does not match the frozen replay version')
    document = json.loads(raw.decode('utf-8-sig'))
    if document.get('schema_version') != 1 or not isinstance(document.get('reviews'), dict):
        raise ValueError('Invalid completed-review document')
    if any(not isinstance(row, dict) or (row.get('review_complete') and
           row.get('decision') not in ('present', 'absent', 'uncertain'))
           for row in document['reviews'].values()):
        raise ValueError('Invalid completed-review decision')
    return document['reviews'], {records_path: digest, configs[0]: config_hash}, {
        'source_reviews_sha256': digest, 'source_project_sha256': config_hash,
        'completed_decisions': dict(Counter(
            row.get('decision') for row in document['reviews'].values()
            if row.get('review_complete'))),
    }


def prepare_curated_training_dataset(replay, output, review_project=None):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(replay) or replay.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the source replay')
    project = Path(review_project).resolve() if review_project is not None else None
    if project and (output.is_relative_to(project) or project.is_relative_to(output)):
        raise ValueError('Output cannot contain or be contained by the source project')
    before = _fingerprints(replay)
    if any(name in before for name in ('CURATED_DATASET_README.txt', 'curated_dataset_metadata.json')):
        raise ValueError('Source is already a curated export')
    rows, _, sizes = _validate_replay(replay, before)
    records, watched, source_summary = _review_source(replay, project)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()
    for relative, digest in before.items():
        source, destination = replay / relative, stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as reader, destination.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        if sha256(destination) != digest:
            raise ValueError('Source changed during copy; staging was not published')
    gt = stage / 'ground_truth'
    gt.mkdir()
    for folder in ('clinician_foreground', 'known_pixels', 'reviewed_masks'):
        (gt / folder).mkdir()
    labels, sparse_index, reviewed_index = [], [], []
    if project:
        sys.path.insert(0, str(Path(__file__).parent / 'painter/src/main/python'))
        from review_utils import reconstruct_review_binary
    for row in rows:
        name, decision, split = row['filename'], row['decision'], row['split']
        definite = decision in ('present', 'absent')
        labels.append(dict(filename=name, patient_id=row['patient_id'], split=split,
                           decision=decision,
                           binary_label=int(decision == 'present') if definite else '',
                           binary_evaluation_eligible=definite,
                           reference_status=row.get('reference_status', ''),
                           independently_adjudicated=False))
        item = dict(filename=name, split=split, decision=decision,
                    reference_type='sparse_corrective_annotation',
                    exhaustive_lesion_reference=False,
                    training_eligible=row['training_eligible'],
                    foreground_mask='', known_pixel_mask='',
                    status=row.get('supervision_exclusion_reason', '') or 'no_eligible_corrections')
        if row['training_eligible'] == 'True':
            annotation_name = Path(name).stem + '.png'
            with Image.open(stage / 'annotations' / split / annotation_name) as image:
                rgba = np.asarray(image.convert('RGBA'))
            red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
            green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
            foreground = 'clinician_foreground/' + annotation_name
            known = 'known_pixels/' + annotation_name
            _png(red, gt / foreground)
            _png(red | green, gt / known)
            item.update(foreground_mask=foreground, known_pixel_mask=known,
                        status='available_sparse_only')
        sparse_index.append(item)
        reviewed = dict(filename=name, split=split, decision=decision, mask='',
                        reference_type='completed_review_reconstruction',
                        independently_adjudicated=False, exhaustive_lesion_reference=False,
                        reconstruction_mode='', segmentation_sha256='', annotation_sha256='',
                        status='no_completed_review_source')
        record = records.get(name, {})
        if not definite:
            reviewed['status'] = decision + '_not_definitive_reference'
        elif project and record.get('review_complete'):
            if record.get('decision') != decision:
                raise ValueError('Review decision conflicts with frozen replay')
            segmentation = _snapshot(project, record, 'segmentation', watched, sizes[name])
            annotation = _snapshot(project, record, 'annotation', watched, sizes[name])
            mode = record.get('reconstruction_mode', 'prediction_corrected')
            mask = reconstruct_review_binary(segmentation, annotation,
                                             decision=decision, reconstruction_mode=mode)
            reviewed.update(reconstruction_mode=mode,
                            segmentation_sha256=record['segmentation_sha256'],
                            annotation_sha256=record.get('annotation_sha256') or '')
            if decision == 'present' and not mask.any():
                reviewed['status'] = 'positive_review_has_empty_mask_needs_correction'
            else:
                relative = 'reviewed_masks/' + Path(name).stem + '.png'
                _png(mask, gt / relative)
                reviewed.update(mask=relative, status='available_completed_review')
        reviewed_index.append(reviewed)
    for filename, entries in (('bscan_labels.csv', labels),
                              ('sparse_mask_index.csv', sparse_index),
                              ('reviewed_mask_index.csv', reviewed_index)):
        write_csv(gt / filename, entries, list(entries[0]))
    result = dict(
        schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='curated_development_dataset', final_test_created=False,
        images=len(rows), decisions=dict(Counter(r['decision'] for r in rows)),
        eligible_sparse_annotations=sum(r['training_eligible'] == 'True' for r in rows),
        reviewed_masks=sum(bool(r['mask']) for r in reviewed_index),
        reviewed_mask_statuses=dict(Counter(r['status'] for r in reviewed_index)),
        source_review_summary=source_summary,
        splits={split: dict(images=sum(r['split'] == split for r in rows),
                            patients=len({r['patient_id'] for r in rows if r['split'] == split}))
                for split in ('train', 'val')},
        source_manifest_sha256=before['manifest.csv'],
        source_replay_metadata_sha256=before['replay_metadata.json'],
    )
    readme = '''RIPL DEVELOPMENT DATASET

Images and all reference files are local. No model was trained by this export.
Use images/ when selecting the image directory in RetinaPainter. Supply the root
manifest.csv for the existing patient assignments. The GUI will not automatically
load reference masks as editable annotations. The offline replay runner can use
this dataset root directly; annotations/train and annotations/val are its sparse
training inputs. Patient roles and original replay files are unchanged.

ground_truth/bscan_labels.csv: recorded present/absent/uncertain/unreviewed scan
decisions. Only present/absent provide definite binary labels. These decisions
have not been independently adjudicated. Legacy red marks alone do not imply a
complete scan-level label.

ground_truth/reviewed_masks/: masks reconstructed from hash-verified completed
review snapshots. Historical prediction_corrected masks include accepted model
suggestions. These are reviewed reference masks, not certified exhaustive lesion
ground truth. See reviewed_mask_index.csv for reconstruction and availability.
A positive review with an empty mask remains positive and has no usable mask;
it is flagged for correction, never relabeled negative. Uncertain and unreviewed
scans have no definitive mask here.

ground_truth/clinician_foreground/ and ground_truth/known_pixels/: binary 0/255
masks derived from eligible sparse red/green corrections. Only known pixels are
supervised. Zero outside known_pixels means unknown, not verified background.
Use sparse_mask_index.csv to identify available masks and withheld corrections.
Do not use reviewed_masks as a new dense training target without a separate,
explicitly evaluated training policy.

Uncertain scans are retained in images and tables, excluded from definitive
binary evaluation and frozen sparse training. Unreviewed legacy scans may have
eligible sparse corrections without a definite scan diagnosis.

This is the existing development cohort, not an untouched final test dataset.
New returned annotations need a new version while retaining patient assignments
and prior exposure. No returned follow-up labels are invented by this export.
Source provenance and exclusions are retained in provenance/ and excluded.csv.
Keep all images and patient-level tables out of assistant/chat/cloud services.

AGGREGATE SUMMARY
'''
    (stage / 'CURATED_DATASET_README.txt').write_text(readme + json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if any(sha256(path) != digest for path, digest in watched.items()) or _fingerprints(replay) != before:
        raise ValueError('Source changed during preparation; staging was not published')
    fingerprints = _fingerprints(stage)
    if any(fingerprints.get(relative) != digest for relative, digest in before.items()):
        raise ValueError('Copied replay differs from source')
    _validate_replay(stage, fingerprints)
    result['file_sha256'] = fingerprints
    (stage / 'curated_dataset_metadata.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if output.exists():
        raise ValueError('Output appeared during preparation; refusing replacement')
    stage.rename(output)
    return {key: value for key, value in result.items() if key != 'file_sha256'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--review-project', type=Path)
    args = parser.parse_args()
    try:
        result = prepare_curated_training_dataset(args.replay, args.output, args.review_project)
    except Exception as error:
        # Exception messages from dependencies may contain patient filenames.
        parser.exit(2, 'Dataset was not published (' + type(error).__name__ + '). '
                    'Inspect sources locally; no clinical details were printed.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
