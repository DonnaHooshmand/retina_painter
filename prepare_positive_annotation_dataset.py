"""Prepare a local positive-candidate annotation session without preloading answers.

Clinical rows and files stay on disk. CLI output contains aggregate counts only.
"""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import unicodedata
from uuid import uuid4

import openpyxl
from PIL import Image

from prepare_curated_training_dataset import sha256, read_manifest, write_csv, _fingerprints


HEADERS = ('Full Path', 'Eye (OD/OS)', 'Image Number', 'Patient ID', 'RIPL', 'Multiple RIPL')
SKIP_DIRS = {'provenance', 'ground_truth', 'ground_truth_binary', 'doctor_foreground_masks',
             'doctor_foreground_overlays', 'doctor_foreground_reference_overlays',
             'ground_truth_audit_overlays', 'models', 'annotations', 'segmentations',
             'reviews', 'review_exports', 'coordinator_only', 'preparation_sources'}
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def norm(value):
    return unicodedata.normalize('NFC', str(value)).strip().casefold()


def safe_name(name):
    return bool(name) and name not in ('.', '..') and not any(c in name for c in '/\\:')


def rank_key(seed, text):
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def project_inventory(syncdir):
    files = set()
    for project in (Path(syncdir) / 'projects').iterdir():
        if project.is_dir() and not project.name.startswith('.'):
            files.update(project.glob('*.seg_proj'))
            files.update((project / 'annotations').rglob('*.png'))
            record = project / 'reviews/review_records.json'
            if record.is_file():
                files.add(record)
    return files


def patient_order(rows, count, seed=0):
    """Round-robin patients, with eight train and two val patients first."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['patient_id']].append(row)
    patients = sorted(groups, key=lambda p: rank_key(seed, 'patient:' + p))
    for patient in patients:
        groups[patient].sort(key=lambda r: rank_key(seed, 'scan:' + r['filename']))
    train = [p for p in patients if groups[p][0]['split'] == 'train']
    val = [p for p in patients if groups[p][0]['split'] == 'val']
    if len(train) < 8 or len(val) < 2 or count < 10:
        raise ValueError('Insufficient patients for the ten-scan training warm-up')
    first = [train[0], val[0], train[1], val[1], *train[2:8]]
    ordered_patients = first + [p for p in patients if p not in first]
    queues = {p: deque(groups[p]) for p in patients}
    selected = []
    while len(selected) < count:
        added = False
        for patient in ordered_patients:
            if queues[patient]:
                selected.append(queues[patient].popleft())
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError('Not enough eligible scans')
    return selected


def discover(workbook, current, assignments, syncdir, count=200, seed=0):
    """Read and validate sources. Does not write clinical data anywhere."""
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError('Seed must be a nonnegative 32-bit integer')
    workbook, current, assignments, syncdir = map(Path, (workbook, current, assignments, syncdir))
    watched = {}

    def watch(path):
        path = Path(path)
        value = sha256(path)
        if path in watched and watched[path] != value:
            raise ValueError('Source changed during preparation')
        watched[path] = value
        return value

    for path in (workbook, current / 'manifest.csv', assignments):
        watch(path)
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    ws = wb['RIPL Annotation']
    if tuple(next(ws.iter_rows(values_only=True))) != HEADERS:
        raise ValueError('Unexpected workbook schema')
    workbook_rows = defaultdict(list)
    for number, row in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
        if row[0]:
            parts = str(row[0]).replace('\\', '/').split('/')
            filename = unicodedata.normalize('NFC', '_'.join(parts[-3:])).strip()
            if len(parts) < 3 or not safe_name(filename):
                raise ValueError('Invalid source image path')
            workbook_rows[norm(filename)].append((number, filename, row))
    wb.close()
    candidates, exclusions = [], []
    for key, entries in workbook_rows.items():
        if not any(e[2][4] == 1 for e in entries):
            continue
        if len(entries) != 1:
            exclusions.append(dict(filename=entries[0][1], reason='ambiguous_workbook_rows'))
            continue
        number, filename, row = entries[0]
        patient = str(row[3]).strip().split('_')[0]
        filename_patient = filename.split('_')[0]
        if not patient.isdecimal() or patient != filename_patient:
            exclusions.append(dict(filename=filename, reason='patient_identity_mismatch'))
            continue
        candidates.append(dict(filename=filename, patient_id=patient, eye=str(row[1]),
                               spreadsheet_row=number, spreadsheet_label=1,
                               spreadsheet_full_path=str(row[0]),
                               reference_status='spreadsheet_positive_candidate_awaiting_review'))

    existing = read_manifest(current / 'manifest.csv', ('filename', 'image_sha256'))
    excluded_stems = {norm(Path(r['filename']).stem) for r in existing}
    excluded_hashes = {r['image_sha256'] for r in existing}
    roles = {}
    for row in read_manifest(assignments, ('patient_id', 'split', 'filename', 'image_sha256')):
        patient, role = row['patient_id'].strip(), row['split']
        if not patient or role not in ('train', 'val') or (patient in roles and roles[patient] != role):
            raise ValueError('Historical patient assignments are ambiguous')
        roles[patient] = role

    project_files = project_inventory(syncdir)
    for path in project_files:
        watch(path)
    missing_review_images = 0
    for project in sorted((syncdir / 'projects').iterdir()):
        if not project.is_dir() or project.name.startswith('.'):
            continue
        annotations = list((project / 'annotations').rglob('*.png'))
        records_path = project / 'reviews/review_records.json'
        reviewed_names = set()
        if records_path.is_file():
            watch(records_path)
            project_files.add(records_path)
            records = json.loads(records_path.read_text(encoding='utf-8-sig'))
            reviewed_names = {name for name, row in records.get('reviews', {}).items()
                              if row.get('review_complete') is True}
        stems = {norm(p.stem) for p in annotations} | {norm(Path(n).stem) for n in reviewed_names}
        excluded_stems.update(stems)
        if not stems:
            continue
        configs = list(project.glob('*.seg_proj'))
        if len(configs) != 1:
            raise ValueError('Previously annotated project lacks an unambiguous config')
        watch(configs[0])
        project_files.add(configs[0])
        config = json.loads(configs[0].read_text(encoding='utf-8-sig'))
        image_dir = (syncdir / 'datasets' / config['dataset']).resolve()
        image_map = defaultdict(list)
        if image_dir.is_dir():
            for path in image_dir.iterdir():
                if path.is_file() and path.suffix.casefold() in EXTENSIONS:
                    image_map[norm(path.stem)].append(path)
        for stem in stems:
            images = image_map.get(stem, [])
            if images:
                for path in images:
                    excluded_hashes.add(watch(path))
            else:
                missing_review_images += 1
        # Track annotation names too, including empty legacy placeholders.
        for path in annotations:
            watch(path)
            project_files.add(path)

    remaining = []
    for row in candidates:
        if norm(Path(row['filename']).stem) in excluded_stems:
            exclusions.append(dict(filename=row['filename'], reason='prior_review_or_annotation'))
        else:
            remaining.append(row)
    keys = {norm(r['filename']) for r in remaining}
    sources = defaultdict(list)
    for root in (syncdir / 'datasets', syncdir / 'curated_datasets'):
        for parent, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d.casefold() not in SKIP_DIRS and not d.startswith('.')]
            for name in files:
                if norm(name) in keys:
                    sources[norm(name)].append(Path(parent) / name)
    valid, content_seen = [], {}
    for row in sorted(remaining, key=lambda r: norm(r['filename'])):
        paths = sorted(set(sources[norm(row['filename'])]), key=lambda p: (len(p.parts), str(p)))
        reason = ''
        if not paths:
            reason = 'source_image_missing'
        else:
            hashes = {watch(p) for p in paths}
            if len(hashes) != 1:
                reason = 'same_filename_different_content'
            else:
                digest = next(iter(hashes))
                if digest in excluded_hashes:
                    reason = 'previously_reviewed_image_content'
                elif digest in content_seen:
                    prior = content_seen[digest]
                    if prior['patient_id'] != row['patient_id'] or prior.get('content_identity_conflict'):
                        prior['content_identity_conflict'] = True
                        if prior in valid:
                            valid.remove(prior)
                            exclusions.append(dict(filename=prior['filename'], reason='same_content_conflicting_patients'))
                        reason = 'same_content_conflicting_patients'
                    else:
                        reason = 'duplicate_image_content'
                else:
                    content_seen[digest] = row
                    with Image.open(paths[0]) as image:
                        image.load()
                        width, height = image.size
                    row.update(source_path=str(paths[0]), image_sha256=digest, width=width, height=height,
                               patient_identity_status='consistent_filename_and_spreadsheet')
                    valid.append(row)
        if reason:
            exclusions.append(dict(filename=row['filename'], reason=reason))
    new_patients = sorted({r['patient_id'] for r in valid} - roles.keys(),
                          key=lambda p: rank_key(seed, 'new-patient-role:' + p))
    new_val = set(new_patients[:math.ceil(len(new_patients) / 5)])
    for row in valid:
        patient = row['patient_id']
        row['split'] = roles.get(patient, 'val' if patient in new_val else 'train')
        row['split_source'] = 'preserved_verified_assignment' if patient in roles else 'new_patient_seeded_development_assignment'
    selected = patient_order(valid, count, seed)
    for rank, row in enumerate(selected, 1):
        row['rank'] = rank
        row['session_batch'] = 1 if rank <= 100 else 2
    train = {r['patient_id'] for r in selected if r['split'] == 'train'}
    val = {r['patient_id'] for r in selected if r['split'] == 'val'}
    if train & val or len({norm(Path(r['filename']).stem) for r in selected}) != count:
        raise ValueError('Selected patient or filename partitions overlap')
    summary = dict(selected_scans=count, selected_patients=len(train | val),
                   eligible_scans=len(valid), eligible_patients=len({r['patient_id'] for r in valid}),
                   first_100_patients=len({r['patient_id'] for r in selected[:100]}),
                   selected_spreadsheet_positive=count, prior_answers_preloaded=False,
                   train_scans=sum(r['split'] == 'train' for r in selected),
                   validation_scans=sum(r['split'] == 'val' for r in selected),
                   train_patients=len(train), validation_patients=len(val), patient_overlap=0,
                   warmup_split=dict(Counter(r['split'] for r in selected[:10])),
                   assignment_sources=dict(Counter(r['split_source'] for r in selected)),
                   exclusions=dict(Counter(r['reason'] for r in exclusions)),
                   unavailable_historical_source_images=missing_review_images,
                   model_type='unet', seed=seed, final_test_created=False)
    if project_inventory(syncdir) != project_files:
        raise ValueError('Project inventory changed during candidate selection')
    return selected, exclusions, summary, watched, project_files


def check_sources(watched):
    if any(not p.is_file() or sha256(p) != digest for p, digest in watched.items()):
        raise ValueError('Source files changed during preparation')


def publish(selected, exclusions, summary, watched, workbook, assignments, syncdir, output, project, project_files=None):
    output, project, syncdir = map(lambda p: Path(p).resolve(), (output, project, syncdir))
    if output.exists() or project.exists() or output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError('Outputs must be new separate directories')
    if not output.is_relative_to(syncdir / 'curated_datasets') or not project.is_relative_to(syncdir / 'projects'):
        raise ValueError('Outputs must use the local curated dataset and project roots')
    check_sources(watched)
    if project_files is not None and project_inventory(syncdir) != project_files:
        raise ValueError('Project inventory changed before publication')
    output.parent.mkdir(parents=True, exist_ok=True)
    project.parent.mkdir(parents=True, exist_ok=True)
    data_stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    proj_stage = project.parent / ('.' + project.name + '.partial-' + uuid4().hex)
    data_stage.mkdir()
    proj_stage.mkdir()
    (data_stage / 'images').mkdir()
    (data_stage / 'ground_truth').mkdir()
    (data_stage / 'coordinator_only').mkdir()
    for row in selected:
        source, target = Path(row['source_path']), data_stage / 'images' / row['filename']
        shutil.copy2(source, target)
        if sha256(target) != row['image_sha256']:
            raise ValueError('Copied image integrity failure')
    fields = ['rank', 'filename', 'patient_id', 'split', 'session_batch'] + [k for k in selected[0] if k not in ('rank', 'filename', 'patient_id', 'split', 'session_batch')]
    write_csv(data_stage / 'manifest.csv', selected, fields)
    write_csv(data_stage / 'coordinator_only/exclusions.csv', exclusions, ('filename', 'reason'))
    patients = [dict(patient_id=p, split=next(r['split'] for r in selected if r['patient_id'] == p),
                     scans=sum(r['patient_id'] == p for r in selected)) for p in sorted({r['patient_id'] for r in selected})]
    write_csv(data_stage / 'patient_assignments.csv', patients, ('patient_id', 'split', 'scans'))
    for folder in ('annotations/train', 'annotations/val', 'segmentations', 'models', 'messages', 'logs', 'reviews', 'review_exports'):
        (proj_stage / folder).mkdir(parents=True)
    config = dict(name=project.name, dataset=(output / 'images').as_posix(),
                  location=project.relative_to(syncdir).as_posix(), original_model_file='random weights',
                  file_names=[r['filename'] for r in selected],
                  train_file_names=[r['filename'] for r in selected if r['split'] == 'train'],
                  val_file_names=[r['filename'] for r in selected if r['split'] == 'val'],
                  split_method='preserved_patient_roles_then_seeded_new_patients_v1',
                  image_order_method='patient_round_robin_priority100_with_8train_2val_warmup_v1',
                  image_order_seed=summary['seed'], training_seed=summary['seed'], model_type='unet',
                  review_schema_version=1, auto_start_training_after_reviews=10)
    (proj_stage / (project.name + '.seg_proj')).write_text(json.dumps(config, indent=2), encoding='utf-8')
    for name in ('manifest.csv', 'patient_assignments.csv'):
        shutil.copyfile(data_stage / name, proj_stage / name)
    readme = f'''FRIDAY POSITIVE-CANDIDATE ANNOTATION SESSION
Session: Friday, October 23, 2026

Open in RetinaPainter using Project > Open project:
{project / (project.name + '.seg_proj')}
Sync directory: {syncdir}
Clean images: {output / 'images'}

There are {summary['selected_scans']} spreadsheet-positive candidate scans across
{summary['selected_patients']} patients. The first 100 are the priority batch and
cover {summary['first_100_patients']} different patients. Continue with the remaining
scans if time allows. The prepared project follows this order automatically.

These are new image annotations, not confirmed positive ground truth. Prior
completed reviews and existing annotation files were excluded conservatively.
Raphael should mark each RIPL location and still select Absent or Uncertain if
he disagrees with the spreadsheet. Avoid painting ambiguous tissue as definite
foreground or background. No masks, reference overlays, answers, or trained
weights are preloaded. Ground truth and reference overlays do not exist yet.

This is a fresh U-Net session (seed {summary['seed']}). The first ten include eight
training and two validation patients. Training starts automatically after those
ten reviews are complete and the user advances, if the trainer is running and
both partitions contain painted annotations. Red marks supervise foreground;
green marks supervise confidently normal background. Positive-only scans can
still contain background pixels. An Absent decision alone supplies no painted
background. Uncertain does not remove painted pixels from the training loss.

Patient assignments are fixed before annotation. Known verified development
assignments are preserved; new patients receive seeded train/validation roles.
No patient occurs in both partitions. These are development scans, not a final
test cohort. A positive-only session cannot measure specificity or prevalence.

At the end, choose Project > Export completed reviews. Exported review masks
are saved under this new project's review_exports folder. A later curated
ground-truth version should be built from the completed review export.

All data remain local. Existing projects and source data were not modified.
manifest.csv contains local source provenance and navigation rank.
coordinator_only/exclusions.csv records selection exclusions.
'''
    for root in (data_stage, proj_stage):
        (root / 'README.txt').write_text(readme, encoding='utf-8')
    (data_stage / 'ground_truth/README.txt').write_text(
        'No ground-truth masks yet. Spreadsheet positivity is a selection label only.\n'
        'Use the new project to collect reviews, then export completed reviews.\n'
        'Do not treat this folder as containing reference answers.\n', encoding='utf-8')
    shutil.copyfile(Path(__file__), data_stage / 'coordinator_only' / Path(__file__).name)
    metadata = dict(summary, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    source_workbook=str(workbook), assignment_source=str(assignments),
                    source_sha256={str(p): h for p,h in watched.items()})
    (data_stage / 'preparation_metadata.json').write_text(
        json.dumps(dict(metadata, file_sha256=_fingerprints(data_stage)), indent=2), encoding='utf-8')
    (proj_stage / 'preparation_metadata.json').write_text(
        json.dumps(dict(summary, file_sha256=_fingerprints(proj_stage)), indent=2), encoding='utf-8')
    check_sources(watched)
    if project_files is not None and project_inventory(syncdir) != project_files:
        raise ValueError('Project inventory changed during publication')
    if output.exists() or project.exists():
        raise ValueError('Output appeared during preparation')
    data_stage.rename(output)
    proj_stage.rename(project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workbook', 'current', 'assignments', 'syncdir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--project', type=Path)
    args = parser.parse_args()
    try:
        if bool(args.output) != bool(args.project):
            raise ValueError('Both output directories are required')
        rows, excluded, summary, watched, project_files = discover(
            args.workbook, args.current, args.assignments, args.syncdir, args.count, args.seed)
        check_sources(watched)
        if args.output:
            publish(rows, excluded, summary, watched, args.workbook, args.assignments,
                    args.syncdir, args.output, args.project, project_files=project_files)
        print(json.dumps(dict(summary, published=bool(args.output), sources_unchanged=True), indent=2))
    except Exception as error:
        parser.exit(2, 'Preparation stopped (' + type(error).__name__ + '); clinical details withheld.\n')


if __name__ == '__main__':
    main()
