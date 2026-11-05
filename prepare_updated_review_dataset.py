"""Merge completed review rounds into a versioned local reference dataset.

Follow-up references supersede earlier reads of identical image bytes. This
creates no training project, model, or new partition assignments. CLI output is
limited to aggregate counts; clinical records and pictures stay on local disk.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np
from PIL import Image

from prepare_curated_training_dataset import _fingerprints, read_manifest, sha256, write_csv
from prepare_practice_review_dataset import _blend, _reference
from review_utils import reconstruct_review_binary, VALID_DECISIONS, VALID_RECONSTRUCTION_MODES


def _safe_name(name):
    return isinstance(name, str) and name not in ('', '.', '..') and not any(c in name for c in '/\\:')


def _contained(root, relative):
    path = root / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)()
                   for p in (path, *path.parents))):
        raise ValueError('Source path is not confined to its root')
    return path


def _watch(path, watched, expected=None):
    found = sha256(path)
    if expected and found != expected:
        raise ValueError('Source hash mismatch')
    if path in watched and watched[path] != found:
        raise ValueError('Source changed while loading')
    watched[path] = found
    return found


def _load_round(root, syncdir, label, watched, manifest=None):
    configs = list(root.glob('*.seg_proj'))
    if len(configs) != 1:
        raise ValueError('Expected one source project configuration')
    record_path = root / 'reviews/review_records.json'
    for path in (configs[0], record_path):
        _watch(path, watched)
    config = json.loads(configs[0].read_text(encoding='utf-8-sig'))
    doc = json.loads(record_path.read_text(encoding='utf-8-sig'))
    if doc.get('schema_version') != 1 or not isinstance(doc.get('reviews'), dict):
        raise ValueError('Unsupported review schema')
    records = doc['reviews']
    order = config.get('file_names', [])
    if (not order or not all(_safe_name(n) for n in order)
            or len({Path(n).stem.casefold() for n in order}) != len(order)
            or set(order) != set(records)):
        raise ValueError('Project order must cover unique completed source images')
    image_dir = (syncdir / 'datasets' / config['dataset']).resolve()
    if manifest is None:
        candidates = [image_dir / 'manifest.csv', image_dir.parent / 'manifest.csv']
        if config.get('dataset_manifest'):
            candidates.insert(0, syncdir / 'datasets' / config['dataset_manifest'])
        manifest = next((p for p in candidates if p.is_file()), None)
    if manifest is None:
        raise ValueError('Explicit patient and image manifest is required')
    manifest = Path(manifest).resolve()
    _watch(manifest, watched)
    manifest_rows = read_manifest(manifest, ('filename', 'patient_id'))
    manifest_by_name = {}
    for row in manifest_rows:
        key = row['filename'].casefold()
        if key in manifest_by_name:
            raise ValueError('Manifest has duplicate image names')
        manifest_by_name[key] = row
    entries, content_seen = [], set()
    source_files = {configs[0].name: configs[0], 'reviews/review_records.json': record_path,
                    'dataset_manifest.csv': manifest}
    state = root / 'reviews/training_state.json'
    if state.is_file():
        _watch(state, watched)
        source_files['reviews/training_state.json'] = state
    for name in order:
        record = records[name]
        if (not isinstance(record, dict) or record.get('review_complete') is not True
                or record.get('decision') not in VALID_DECISIONS
                or record.get('reconstruction_mode', 'prediction_corrected') not in VALID_RECONSTRUCTION_MODES):
            raise ValueError('Every source review must be completed with a valid decision')
        mapping = manifest_by_name.get(name.casefold())
        if not mapping:
            raise ValueError('Image is missing from source manifest')
        expected_hash = mapping.get('sha256') or mapping.get('image_sha256')
        if not expected_hash:
            raise ValueError('Source manifest image hash is required')
        image_path = _contained(image_dir, name)
        image_hash = _watch(image_path, watched, expected_hash)
        if image_hash in content_seen:
            raise ValueError('Duplicate image content within one review round')
        content_seen.add(image_hash)
        with Image.open(image_path) as image:
            image.load()
            size = image.size
        paths = {}
        for field, hash_field, required in (
                ('segmentation_snapshot', 'segmentation_sha256', True),
                ('annotation_snapshot', 'annotation_sha256', False),
                ('decision_preview', 'decision_preview_sha256', False)):
            relative = record.get(field)
            if not relative:
                if required:
                    raise ValueError('Required review snapshot missing')
                paths[field] = None
                continue
            if not record.get(hash_field):
                raise ValueError('Review snapshot hash missing')
            path = _contained(root, relative)
            _watch(path, watched, record[hash_field])
            if field != 'decision_preview':
                with Image.open(path) as image:
                    if image.size != size:
                        raise ValueError('Review snapshot dimensions differ from image')
            paths[field] = path
            source_files[Path(relative).as_posix()] = path
        roles = [role for role, key in [('train', 'train_file_names'), ('val', 'val_file_names')]
                 if name in config.get(key, [])]
        entries.append(dict(source_filename=name, round=label, record=record,
                            image=image_path, image_sha256=image_hash, mapping=mapping,
                            source_train_val_role=';'.join(roles) or 'unassigned',
                            paths=paths))
    return entries, source_files


def _patient_identity(versions, audit):
    ids = {v['mapping']['patient_id'].strip() for v in versions if v['mapping']['patient_id'].strip()}
    if audit and audit.get('patient_id', '').strip():
        ids.add(audit['patient_id'].strip())
    source_conflict = any('conflict' in v['mapping'].get('patient_identity_status', '').lower()
                          for v in versions)
    if len(ids) > 1 or source_conflict:
        return '', 'conflict_hold'
    if audit is not None and not audit.get('patient_id', '').strip():
        return '', 'identity_hold'
    if not ids:
        return '', 'missing_hold'
    if audit is not None:
        return next(iter(ids)), 'previously_mapped'
    if any(v['mapping'].get('patient_identity_status') in ('consistent', 'verified', 'confirmed')
           for v in versions):
        return next(iter(ids)), 'consistent_source_manifest'
    return next(iter(ids)), 'supplied_unverified'


def _copy(source, destination, expected):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer)
    if sha256(destination) != expected:
        raise ValueError('Copied file hash mismatch')


def _check_frozen_export(root, entries, watched):
    """Check the frozen export's meaning as well as its file fingerprints."""
    export = root / 'completed_reviews'
    csv_path = export / 'reviews.csv'
    _watch(csv_path, watched)
    rows = read_manifest(csv_path, ('filename', 'decision', 'binary_mask'))
    by_name = {row['filename']: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != {e['source_filename'] for e in entries}:
        raise ValueError('Frozen export and source reviews cover different images')
    for entry in entries:
        row, record = by_name[entry['source_filename']], entry['record']
        mode = record.get('reconstruction_mode', 'prediction_corrected')
        fields = dict(decision=record['decision'], reconstruction_mode=mode,
                      review_complete='True',
                      segmentation_snapshot=record['segmentation_snapshot'],
                      segmentation_sha256=record['segmentation_sha256'],
                      annotation_snapshot=record.get('annotation_snapshot') or '',
                      annotation_sha256=record.get('annotation_sha256') or '')
        if any(row.get(key) != value for key, value in fields.items()):
            raise ValueError('Frozen export disagrees with saved review')
        mask_path = _contained(export, row['binary_mask'])
        if not row.get('binary_mask_sha256'):
            raise ValueError('Frozen export mask hash missing')
        _watch(mask_path, watched, row['binary_mask_sha256'])
        with Image.open(mask_path) as image:
            saved_mask = np.array(image)
        reconstructed = reconstruct_review_binary(
            entry['paths']['segmentation_snapshot'], entry['paths']['annotation_snapshot'],
            decision=record['decision'], reconstruction_mode=mode)
        if (not np.array_equal(saved_mask, reconstructed.astype(np.uint8) * 255)
                or row.get('foreground_pixels') != str(int(reconstructed.sum()))):
            raise ValueError('Frozen mask disagrees with saved review')


def build_updated_dataset(original_project, frozen_followup, syncdir, output,
                          original_manifest=None, identity_audit=None,
                          expected_original_reviews_sha256=None):
    original_project, frozen_followup, syncdir, output = [Path(p).resolve() for p in
                                                        (original_project, frozen_followup, syncdir, output)]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (original_project, frozen_followup)):
        raise ValueError('Output must be new and separate from sources')
    frozen_before = _fingerprints(frozen_followup)
    metadata = json.loads((frozen_followup / 'freeze_metadata.json').read_text(encoding='utf-8-sig'))
    expected = metadata.get('file_sha256', {})
    if (not expected or set(frozen_before) != set(expected) | {'freeze_metadata.json'}
            or any(frozen_before.get(name) != digest for name, digest in expected.items())):
        raise ValueError('Frozen follow-up integrity check failed')
    watched = {}
    if expected_original_reviews_sha256:
        _watch(original_project / 'reviews/review_records.json', watched,
               expected_original_reviews_sha256)
    original, original_files = _load_round(original_project, syncdir, 'original', watched, original_manifest)
    followup, _ = _load_round(frozen_followup / 'source_snapshot', syncdir, 'followup', watched,
                              frozen_followup / 'source_snapshot/dataset_manifest.csv')
    _check_frozen_export(frozen_followup, followup, watched)
    for entry in original + followup:
        if output.is_relative_to(entry['image'].parent) or entry['image'].is_relative_to(output):
            raise ValueError('Output overlaps source images')
    audit_by_hash = {}
    if identity_audit:
        identity_audit = Path(identity_audit).resolve()
        _watch(identity_audit, watched)
        for row in read_manifest(identity_audit, ('image_sha256', 'patient_id')):
            if row['image_sha256'] in audit_by_hash:
                raise ValueError('Ambiguous prior identity audit')
            audit_by_hash[row['image_sha256']] = row

    by_hash, canonical_order, name_hashes = {}, [], {}
    for entry in original + followup:
        name, digest = entry['source_filename'], entry['image_sha256']
        for key in (name.casefold(), Path(name).stem.casefold()):
            if key in name_hashes and name_hashes[key] != digest:
                raise ValueError('Same image name refers to different content')
            name_hashes[key] = digest
        if digest not in by_hash:
            by_hash[digest] = dict(filename=name, versions=[])
            canonical_order.append(digest)
        by_hash[digest]['versions'].append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()
    folders = ('images', 'ground_truth', 'ground_truth_binary', 'doctor_foreground_masks',
               'doctor_foreground_overlays', 'doctor_foreground_reference_overlays',
               'ground_truth_audit_overlays')
    for folder in folders:
        (stage / folder).mkdir()
    for relative, source in original_files.items():
        _copy(source, stage / 'provenance/original' / relative, watched[source])
    for relative, digest in frozen_before.items():
        _copy(frozen_followup / relative, stage / 'provenance/followup' / relative, digest)
    if identity_audit:
        _copy(identity_audit, stage / 'provenance/prior_identity_audit.csv', watched[identity_audit])

    rows, history = [], []
    for rank, digest in enumerate(canonical_order, 1):
        item = by_hash[digest]
        name, versions = item['filename'], item['versions']
        current = versions[-1]
        record = current['record']
        decision = record['decision']
        mode = record.get('reconstruction_mode', 'prediction_corrected')
        _copy(current['image'], stage / 'images' / name, digest)
        with Image.open(current['image']) as image:
            rgb = np.array(image.convert('RGB'))
        annotation = current['paths']['annotation_snapshot']
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        if annotation:
            with Image.open(annotation) as image:
                rgba = np.array(image.convert('RGBA'))
        red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
        green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
        if (red & green).any() or (decision == 'absent' and red.any()):
            raise ValueError('Contradictory current correction marks')
        mask = reconstruct_review_binary(current['paths']['segmentation_snapshot'], annotation,
                                         decision=decision, reconstruction_mode=mode)
        status = 'completed_review_reference'
        if decision == 'uncertain':
            status = 'uncertain_not_definitive_reference'
        elif decision == 'present' and not mask.any():
            status = 'positive_review_has_empty_mask_needs_correction'
        target = Path(name).stem + '.png'
        Image.fromarray(red.astype(np.uint8) * 255).save(stage / 'doctor_foreground_masks' / target)
        Image.fromarray(_blend(rgb, red, (255, 32, 32), 0.8)).save(stage / 'doctor_foreground_overlays' / target)
        doctor_reference = 'doctor_foreground_reference_overlays/' + target
        reference = 'ground_truth_audit_overlays/' + target
        _reference(rgb, red, mask & ~red, decision, status).save(stage / doctor_reference)
        _reference(rgb, red, mask & ~red, decision, status, audit=True).save(stage / reference)
        rgba_mask, binary_mask = '', ''
        if status == 'completed_review_reference':
            rgba_mask, binary_mask = 'ground_truth/' + target, 'ground_truth_binary/' + target
            corrected = np.zeros((*mask.shape, 4), dtype=np.uint8)
            corrected[mask] = (0, 255, 255, 178)
            Image.fromarray(corrected).save(stage / rgba_mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(stage / binary_mask)
        audit = audit_by_hash.get(digest)
        patient_id, identity_status = _patient_identity(versions, audit)
        prior = versions[0]['record']['decision'] if len(versions) > 1 else ''
        prior_split = audit.get('split', '') if audit else ''
        rows.append(dict(rank=rank, filename=name, image_sha256=digest, patient_id=patient_id,
                         patient_identity_status=identity_status, current_round=current['round'],
                         decision=decision, prior_decision=prior,
                         decision_changed=bool(prior and prior != decision),
                         binary_label=int(decision == 'present') if decision != 'uncertain' else '',
                         binary_evaluation_label_available=decision != 'uncertain',
                         binary_mask=binary_mask, ground_truth_mask=rgba_mask,
                         reference_picture=reference, doctor_reference_picture=doctor_reference,
                         reference_status=status,
                         reconstruction_mode=mode, independently_adjudicated=False,
                         prior_development_split=prior_split if prior_split in ('train', 'val') else '',
                         partition_status='not_assigned_in_this_version'))
        for version in versions:
            rec = version['record']
            prefix = 'provenance/original/' if version['round'] == 'original' else 'provenance/followup/source_snapshot/'
            history.append(dict(filename=name, source_filename=version['source_filename'],
                                round=version['round'], decision=rec['decision'],
                                selected_current_reference=version is current, image_sha256=digest,
                                patient_id=version['mapping']['patient_id'],
                                source_patient_identity_status=version['mapping'].get('patient_identity_status', ''),
                                source_train_val_role=version['source_train_val_role'],
                                completed_at_utc=rec.get('completed_at_utc', ''),
                                segmentation_snapshot=prefix + rec['segmentation_snapshot'],
                                segmentation_sha256=rec['segmentation_sha256'],
                                annotation_snapshot=prefix + rec['annotation_snapshot'] if rec.get('annotation_snapshot') else '',
                                annotation_sha256=rec.get('annotation_sha256') or '',
                                reconstruction_mode=rec.get('reconstruction_mode', 'prediction_corrected')))
    for filename, values in [('manifest.csv', rows), ('ground_truth_status.csv', rows),
                             ('review_history.csv', history)]:
        write_csv(stage / filename, values, list(values[0]))
    summary = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
                   images=len(rows), original_reviews=len(original), followup_reviews=len(followup),
                   repeated_images=sum(len(v['versions']) > 1 for v in by_hash.values()),
                   additional_images=len(rows) - len(original),
                   decisions=dict(Counter(r['decision'] for r in rows)),
                   changed_decisions=sum(r['decision_changed'] for r in rows),
                   current_reference_rounds=dict(Counter(r['current_round'] for r in rows)),
                   review_history_records=len(history), reference_pictures=len(rows),
                   definitive_review_masks=sum(bool(r['binary_mask']) for r in rows),
                   identity_status_counts=dict(Counter(r['patient_identity_status'] for r in rows)),
                   partition_assignments_created=False, training_project_created=False,
                   final_test_created=False,
                   merge_policy='followup_supersedes_original_for_identical_image_sha256')
    readme = '''UPDATED RAPHAEL REFERENCE DATASET

images/: clean original scans, each unique image included once.
ground_truth_audit_overlays/: matching reference pictures showing the current
decision, red doctor marks and cyan retained model regions. Open these beside
the clean images to see the full reconstructed review, including regions that
the doctor accepted without painting another correction.
doctor_foreground_reference_overlays/: matching reference pictures showing the
current recorded decision and the doctor's sparse red RIPL marks.
doctor_foreground_overlays/: the same red overlay without the explanatory footer.
doctor_foreground_masks/: binary masks of explicit red marks only; untouched
pixels are unknown, not confirmed background.
ground_truth/ and ground_truth_binary/: usable reconstructed definite review
masks in RGBA and 0/255 formats. They are clinician-reviewed references and may
include accepted model predictions; they are not independently adjudicated
exhaustive contours. Use the audit overlay to distinguish the two sources.

The frozen follow-up supersedes the earlier reading only when image SHA-256
matches exactly. The original filename remains for repeated scans. New scans
are added once. A changed decision is recorded, not silently erased: all original
and follow-up readings remain in review_history.csv and provenance/.

Uncertain decisions remain UNCERTAIN. They have reference pictures but no
definitive binary mask or binary diagnosis. An empty positive reconstruction
also has no definitive mask; its positive decision is preserved and flagged.
ground_truth_status.csv identifies the current references and their availability.

manifest.csv contains explicit supplied patient IDs and identity status.
Conflicting or previously held identities have a blank active patient_id and
an explicit hold status; their original IDs remain in source-specific history.
prior_development_split and source_train_val_role are historical provenance,
not new train/validation assignments. No new partitions, project, annotations,
or model were created. Establish the patient split in the next step before
training; the current GUI's New Project does not import patient assignments.
This development collection is not an untouched final test cohort.

provenance/original/ preserves the source original reviews, project configuration,
manifest and snapshots. provenance/followup/ preserves the entire frozen export.
These are audit copies, not projects to reopen for editing.
dataset_metadata.json contains hashes of every other file in this bundle.
Keep this version unchanged; subsequent updates need a new directory.
All clinical images and records stay local and out of assistant/cloud services.

AGGREGATE COUNTS
'''
    (stage / 'README.txt').write_text(readme + json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    implementation = stage / 'provenance/preparation_code'
    implementation.mkdir()
    for source in (Path(__file__), Path(__file__).with_name('prepare_practice_review_dataset.py'),
                   Path(__file__).with_name('prepare_curated_training_dataset.py'),
                   Path(__file__).parent / 'painter/src/main/python/review_utils.py'):
        shutil.copyfile(source, implementation / source.name)
    if any(sha256(p) != digest for p, digest in watched.items()) or _fingerprints(frozen_followup) != frozen_before:
        raise ValueError('Sources changed while preparing dataset')
    fingerprints = _fingerprints(stage)
    (stage / 'dataset_metadata.json').write_text(
        json.dumps({**summary, 'file_sha256': fingerprints}, indent=2), encoding='utf-8')
    if output.exists() or stage.resolve().parent != output.parent:
        raise ValueError('Output path changed')
    stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-project', 'frozen-followup', 'syncdir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--original-manifest', type=Path)
    parser.add_argument('--identity-audit', type=Path)
    parser.add_argument('--expected-original-reviews-sha256')
    args = parser.parse_args()
    try:
        result = build_updated_dataset(**vars(args))
    except Exception as error:
        parser.exit(2, 'Dataset was not published (' + type(error).__name__ + ').\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
