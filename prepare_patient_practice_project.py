"""Prepare a blank local annotation project with preserved patient assignments.

The immutable reference dataset is read-only. No images, prior answers, or
trained weights are copied into the project. CLI output is aggregate-only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from prepare_curated_training_dataset import _fingerprints, read_manifest, sha256, write_csv


def _safe_name(name):
    return isinstance(name, str) and name not in ('', '.', '..') and not any(c in name for c in '/\\:')


def _warmup_order(rows):
    """Use an explicit label-informed warm-up; retain source order afterwards."""
    pools = {}
    for split in ('train', 'val'):
        for decision in ('present', 'absent'):
            pool = [r for r in rows if r['split'] == split and r['decision'] == decision]
            if decision == 'present':
                first, repeated, seen = [], [], set()
                for row in pool:
                    (repeated if row['patient_id'] in seen else first).append(row)
                    seen.add(row['patient_id'])
                pool = first + repeated
            pools[split, decision] = pool
    schedule = [('train', 'present'), ('val', 'present'), ('train', 'absent'),
                ('val', 'absent'), ('train', 'present'), ('val', 'present'),
                ('train', 'absent'), ('train', 'absent'), ('val', 'absent'),
                ('train', 'absent')]
    warmup = []
    for key in schedule:
        if pools[key]:
            warmup.append(pools[key].pop(0))
    selected = {r['filename'] for r in warmup}
    for candidates in ([r for r in rows if r['decision'] != 'uncertain'], rows):
        for row in candidates:
            if len(warmup) >= min(10, len(rows)):
                break
            if row['filename'] not in selected:
                warmup.append(row)
                selected.add(row['filename'])
    return warmup + [r for r in rows if r['filename'] not in selected]


def build_patient_practice_project(dataset, assignments, syncdir, output,
                                   model_type='unet', seed=0):
    dataset, assignments, syncdir, output = [Path(p).resolve() for p in
                                            (dataset, assignments, syncdir, output)]
    if model_type not in ('unet', 'retfound', 'retfound_rfa'):
        raise ValueError('Unsupported model type')
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('Trial seed must be a nonnegative 32-bit integer')
    if (output.exists() or output.is_relative_to(dataset) or dataset.is_relative_to(output)
            or assignments.is_relative_to(output)):
        raise ValueError('Project output must be new and separate from source data')
    before = _fingerprints(dataset)
    metadata = json.loads((dataset / 'dataset_metadata.json').read_text(encoding='utf-8-sig'))
    expected = metadata.get('file_sha256', {})
    if (not expected or set(before) != set(expected) | {'dataset_metadata.json'}
            or any(before.get(name) != value for name, value in expected.items())):
        raise ValueError('Reference dataset integrity check failed')
    assignment_hash = sha256(assignments)
    assignment_rows = read_manifest(assignments, ('filename', 'image_sha256', 'patient_id', 'split'))
    patient_roles, content_roles = {}, {}
    for row in assignment_rows:
        patient, split, digest = row['patient_id'].strip(), row['split'], row['image_sha256']
        if not patient or split not in ('train', 'val') or not digest:
            raise ValueError('Verified assignments must have explicit patients and partitions')
        if patient in patient_roles and patient_roles[patient] != split:
            raise ValueError('Verified patient assignments overlap')
        if digest in content_roles and content_roles[digest] != (patient, split):
            raise ValueError('Verified image assignments conflict')
        patient_roles[patient] = split
        content_roles[digest] = (patient, split)
    source_rows = read_manifest(dataset / 'manifest.csv', (
        'rank', 'filename', 'image_sha256', 'patient_id', 'patient_identity_status',
        'decision', 'prior_development_split', 'reference_picture'))
    if not source_rows:
        raise ValueError('Reference dataset is empty')
    source_rows.sort(key=lambda r: int(r['rank']))
    if len({int(r['rank']) for r in source_rows}) != len(source_rows):
        raise ValueError('Source ranks must be unique')
    names, stems, hashes, rows = set(), set(), set(), []
    for source in source_rows:
        name, digest, patient = source['filename'], source['image_sha256'], source['patient_id'].strip()
        if (not _safe_name(name) or name.casefold() in names
                or Path(name).stem.casefold() in stems or digest in hashes):
            raise ValueError('Reference filenames or image contents are ambiguous')
        names.add(name.casefold())
        stems.add(Path(name).stem.casefold())
        hashes.add(digest)
        if before.get('images/' + name) != digest:
            raise ValueError('Image does not match its manifest hash')
        reference = Path(source['reference_picture'])
        if (reference.is_absolute() or '..' in reference.parts
                or reference.as_posix() not in expected):
            raise ValueError('Reference picture is not a verified dataset file')
        if source['decision'] not in ('present', 'absent', 'uncertain'):
            raise ValueError('Unsupported reference decision')
        split, hold_reason = 'hold', 'unresolved_patient_identity'
        if patient:
            if source['patient_identity_status'] not in ('previously_mapped', 'consistent_source_manifest'):
                raise ValueError('Active patient identity is not verified')
            if digest not in content_roles or content_roles[digest][0] != patient:
                raise ValueError('Reference image lacks a matching verified patient assignment')
            split = patient_roles[patient]
            if source['prior_development_split'] not in ('', split):
                raise ValueError('Existing patient partition would change')
            hold_reason = ''
        rows.append(dict(source_rank=int(source['rank']), filename=name, image_sha256=digest,
                         patient_id=patient, patient_identity_status=source['patient_identity_status'],
                         split=split, reference_decision=source['decision'], decision=source['decision'],
                         reference_picture=source['reference_picture'],
                         hold_reason=hold_reason, navigation_rank=''))
    active = _warmup_order([r for r in rows if r['split'] in ('train', 'val')])
    if not active or not all(any(r['split'] == s for r in active) for s in ('train', 'val')):
        raise ValueError('Both training and validation must contain mapped images')
    for rank, row in enumerate(active, 1):
        row['navigation_rank'] = rank
    patient_rows = [dict(patient_id=p, split=s,
                         images=sum(r['patient_id'] == p for r in active))
                    for p, s in sorted({(r['patient_id'], r['split']) for r in active})]
    train_patients = {r['patient_id'] for r in patient_rows if r['split'] == 'train'}
    val_patients = {r['patient_id'] for r in patient_rows if r['split'] == 'val'}
    if train_patients & val_patients:
        raise ValueError('Prepared patient partitions overlap')
    # This bundle sits outside syncdir/datasets. Older painter clients compare
    # current paths to absolute navigation paths without normalizing '..'.
    # A canonical absolute local path works with those clients as well.
    info = dict(name=output.name,
                dataset=(dataset / 'images').resolve().as_posix(),
                location=Path(os.path.relpath(output, syncdir)).as_posix(),
                original_model_file='random weights', file_names=[r['filename'] for r in active],
                train_file_names=[r['filename'] for r in active if r['split'] == 'train'],
                val_file_names=[r['filename'] for r in active if r['split'] == 'val'],
                split_method='preserved_verified_patient_assignments_v1',
                image_order_method='reference_label_warmup_then_source_rank_v1',
                image_order_seed=seed, training_seed=seed, model_type=model_type,
                review_schema_version=1, auto_start_training_after_reviews=10)
    summary = dict(schema_version=1, prepared_at_utc=datetime.now(timezone.utc).isoformat(),
                   dataset_images=len(rows), active_images=len(active), held_images=len(rows) - len(active),
                   splits={s:dict(images=sum(r['split'] == s for r in active),
                                  patients=sum(r['split'] == s for r in patient_rows),
                                  decisions=dict(Counter(r['decision'] for r in active if r['split'] == s)))
                           for s in ('train', 'val')},
                   patient_overlap=0, model_type=model_type, training_seed=seed,
                   warmup={s:dict(Counter(r['decision'] for r in active[:10] if r['split'] == s))
                           for s in ('train', 'val')},
                   warmup_order_is_label_informed=True, annotations_preloaded=False,
                   previous_model_loaded=False, training_started=False, final_test_created=False,
                   reference_dataset_metadata_sha256=before['dataset_metadata.json'],
                   verified_assignments_sha256=assignment_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()
    for folder in ('annotations/train', 'annotations/val', 'segmentations', 'models',
                   'messages', 'logs', 'reviews', 'review_exports', 'preparation_sources'):
        (stage / folder).mkdir(parents=True)
    (stage / (output.name + '.seg_proj')).write_text(json.dumps(info, indent=2), encoding='utf-8')
    fields = list(rows[0])
    write_csv(stage / 'manifest.csv', rows, fields)
    write_csv(stage / 'held_images.csv', [r for r in rows if r['split'] == 'hold'], fields)
    write_csv(stage / 'patient_assignments.csv', patient_rows, ('patient_id', 'split', 'images'))
    for source, name in ((assignments, 'verified_assignment_manifest.csv'),
                         (dataset / 'manifest.csv', 'reference_dataset_manifest.csv'),
                         (dataset / 'dataset_metadata.json', 'reference_dataset_metadata.json')):
        destination = stage / 'preparation_sources' / name
        shutil.copyfile(source, destination)
        expected_hash = assignment_hash if source == assignments else before[source.relative_to(dataset).as_posix()]
        if sha256(destination) != expected_hash:
            raise ValueError('Provenance copy differs from its source')
    for source in (Path(__file__), Path(__file__).with_name('prepare_curated_training_dataset.py')):
        shutil.copyfile(source, stage / 'preparation_sources' / source.name)
    readme = f'''FRESH PATIENT-SEPARATED ANNOTATION PROJECT

OPEN THIS PROJECT
In RetinaPainter choose Project > Open project and select:
{output / (output.name + '.seg_proj')}
Keep the sync directory set to {syncdir}.
Do not recreate this through New Project: that dialog splits individual files,
whereas this prepared project fixes patient-separated train/validation lists.

The project opens clean scans directly from:
{dataset / 'images'}
The project stores this absolute local image path for compatibility with the
current painter's navigation. Keep the reference dataset at this location.
Open matching reference pictures alongside RetinaPainter from:
{dataset / 'ground_truth_audit_overlays'}
Red means doctor-painted foreground; cyan means retained model foreground.
The reference answers are not preloaded as editable annotations or decisions.

PATIENT SPLIT
The previously verified patient assignments are preserved, including the
additional scans. No patient occurs in both train and validation. The current
GUI saves every new annotation to its assigned folder without moving the scan
when you skip, revisit, or complete it. Two files with the same patient always
have the same partition. held_images.csv lists unresolved identities excluded
from navigation; their images and references remain in the immutable dataset.
manifest.csv and patient_assignments.csv document this project's assignments.
Their reference_decision/decision fields are reference metadata only; the GUI
does not read them as your new completed decisions.

START ANNOTATING
This starts a fresh {model_type} project, training seed {seed}, without a prior
trained checkpoint. No annotations, completed reviews, or predictions were
preloaded. The first ten scans intentionally mix positives and negatives,
including positives in both train and validation. The rest retain source rank.
This is a label-informed practice warm-up, not a random evaluation sequence.

Complete each review in order. Paint red on RIPL foreground and green on clear
background as appropriate; an Absent decision alone adds no supervised pixels.
Training is configured to start automatically after the first ten reviews are
complete and you advance. The trainer must be running and both partitions need
painted annotations. Opening the project does not start optimization.

UNCERTAIN remains available. Do not paint ambiguous regions as definite red or
green: any painted pixels are supervised even on a review marked uncertain.
Keep ambiguous regions untouched and record Uncertain when appropriate.

This is a development/practice project with internal validation. The sparse
validation signal is not a final clinical test result, and these patients do
not form a new untouched test cohort. Ground-truth references stay unchanged.
Use the same patient assignments for later model comparisons. Do not copy
reference masks into the editable annotation folders.

preparation_metadata.json records initial file hashes and aggregate counts.
Project files and annotations will naturally change as you work. Preparation
source copies are audit records only. All clinical files remain local.

INITIAL AGGREGATE COUNTS
'''
    (stage / 'README.txt').write_text(readme + json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if _fingerprints(dataset) != before or sha256(assignments) != assignment_hash:
        raise ValueError('Sources changed while preparing project')
    summary_with_hashes = {**summary, 'file_sha256': _fingerprints(stage)}
    (stage / 'preparation_metadata.json').write_text(json.dumps(summary_with_hashes, indent=2), encoding='utf-8')
    if output.exists() or stage.resolve().parent != output.parent:
        raise ValueError('Project output path changed')
    stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'assignments', 'syncdir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model-type', choices=('unet', 'retfound', 'retfound_rfa'), default='unet')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    try:
        result = build_patient_practice_project(**vars(args))
    except Exception as error:
        parser.exit(2, 'Project was not published (' + type(error).__name__ + ').\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
