"""Create clean practice images and matching local doctor reference pictures.

Real images stay on disk. CLI output contains aggregate counts only.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_curated_training_dataset import (
    _fingerprints, _review_source, _snapshot, _validate_replay, sha256, write_csv)

sys.path.insert(0, str(Path(__file__).parent / 'painter/src/main/python'))
from review_utils import reconstruct_review_binary


def _blend(rgb, selected, color, opacity):
    result = np.array(rgb, dtype=np.float32, copy=True)
    result[selected] = result[selected] * (1 - opacity) + np.array(color) * opacity
    return np.rint(result).astype(np.uint8)


def _reference(rgb, red, inherited, decision, status, audit=False):
    """Add a footer outside the native scan; never paint text over anatomy."""
    if audit:
        rgb = _blend(rgb, inherited, (0, 255, 255), 0.55)
    rgb = _blend(rgb, red, (255, 32, 32), 0.80)
    width, height = max(rgb.shape[1], 840), rgb.shape[0]
    output = Image.new('RGB', (width, height + 130), (20, 20, 20))
    output.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(output)
    try:
        font = ImageFont.truetype('arial.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    text = ['Raphael review: RIPL ' + decision.upper(),
            'Red: doctor-painted RIPL marks. Green corrections are not RIPL targets.']
    if audit:
        text.append('Cyan: retained model foreground; red: doctor-painted foreground.')
    else:
        text.append('Use this reference beside the clean image while making your own annotations.')
    if decision == 'uncertain':
        text.append('UNCERTAIN: preserve uncertainty; no definitive lesion mask supplied.')
    elif status == 'positive_review_has_empty_mask_needs_correction':
        text.append('PRESENT decision, but no recorded RIPL location. Needs doctor clarification.')
    elif decision == 'absent':
        text.append('ABSENT: the completed review reports no RIPL on this scan.')
    elif not red.any():
        text.append('No red correction was recorded. Consult the audit overlay for accepted model regions.')
    else:
        text.append('Marks are sparse location references, not precise exhaustive contours.')
    for index, line in enumerate(text):
        draw.text((10, height + 9 + 28 * index), line, fill=(255, 255, 255), font=font)
    return output


def build_practice_dataset(project, replay, syncdir, output, model_type='unet'):
    project, replay, syncdir, output = [Path(p).resolve() for p in (project, replay, syncdir, output)]
    if model_type not in ('unet', 'retfound', 'retfound_rfa'):
        raise ValueError('Unsupported model type')
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (project, replay)):
        raise ValueError('Output must be new and separate from source data')
    frozen = _fingerprints(replay)
    replay_rows, _, _ = _validate_replay(replay, frozen)
    mapping = {row['filename']: row for row in replay_rows}
    records, watched, _ = _review_source(replay, project)
    config = json.loads(next(project.glob('*.seg_proj')).read_text(encoding='utf-8-sig'))
    image_dir = (syncdir / 'datasets' / config['dataset']).resolve()
    if not image_dir.is_dir() or output.is_relative_to(image_dir) or image_dir.is_relative_to(output):
        raise ValueError('Image directory is missing or overlaps output')
    ordered = [name for name in config['file_names'] if records.get(name, {}).get('review_complete')]
    if len(ordered) != sum(bool(r.get('review_complete')) for r in records.values()):
        raise ValueError('Project order does not cover every completed review')
    if not ordered or len(set(n.casefold() for n in ordered)) != len(ordered):
        raise ValueError('No reviews or duplicate image names')
    if len({Path(n).stem.casefold() for n in ordered}) != len(ordered):
        raise ValueError('Ambiguous mask names')
    if any('/' in n or '\\' in n or ':' in n or n in ('.', '..') for n in ordered):
        raise ValueError('Image names must be flat filenames')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()
    folders = ('images', 'ground_truth', 'ground_truth_binary',
               'doctor_foreground_masks', 'doctor_foreground_overlays',
               'doctor_foreground_reference_overlays', 'ground_truth_audit_overlays',
               'provenance/annotation_snapshots', 'provenance/segmentation_snapshots')
    for folder in folders:
        (stage / folder).mkdir(parents=True)
    rows = []
    for rank, name in enumerate(ordered, 1):
        record, reference = records[name], mapping.get(name)
        source = image_dir / name
        if source.is_symlink() or not source.is_file():
            raise ValueError('Source image unavailable')
        image_hash = sha256(source)
        if reference and (image_hash != reference['image_sha256'] or record['decision'] != reference['decision']):
            raise ValueError('Image or decision differs from patient-mapped replay')
        watched[source] = image_hash
        with Image.open(source) as image:
            rgb = np.array(image.convert('RGB'))
            size = image.size
        segmentation = _snapshot(project, record, 'segmentation', watched, size)
        annotation = _snapshot(project, record, 'annotation', watched, size)
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        if annotation:
            with Image.open(annotation) as image:
                rgba = np.asarray(image.convert('RGBA'))
        red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
        green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
        if (red & green).any() or (record['decision'] == 'absent' and red.any()):
            raise ValueError('Contradictory doctor corrections')
        mode = record.get('reconstruction_mode', 'prediction_corrected')
        mask = reconstruct_review_binary(segmentation, annotation, decision=record['decision'],
                                         reconstruction_mode=mode)
        decision = record['decision']
        status = 'completed_review_reference'
        if decision == 'uncertain':
            status = 'uncertain_not_definitive_reference'
        elif decision == 'present' and not mask.any():
            status = 'positive_review_has_empty_mask_needs_correction'
        target_name = Path(name).stem + '.png'
        shutil.copyfile(source, stage / 'images' / name)
        if sha256(stage / 'images' / name) != image_hash:
            raise ValueError('Image copy mismatch')
        shutil.copyfile(segmentation, stage / 'provenance/segmentation_snapshots' / target_name)
        if sha256(stage / 'provenance/segmentation_snapshots' / target_name) != record['segmentation_sha256']:
            raise ValueError('Segmentation snapshot copy mismatch')
        if annotation:
            shutil.copyfile(annotation, stage / 'provenance/annotation_snapshots' / target_name)
            if sha256(stage / 'provenance/annotation_snapshots' / target_name) != record['annotation_sha256']:
                raise ValueError('Annotation snapshot copy mismatch')
        Image.fromarray(red.astype(np.uint8) * 255).save(stage / 'doctor_foreground_masks' / target_name)
        Image.fromarray(_blend(rgb, red, (255, 32, 32), 0.8)).save(
            stage / 'doctor_foreground_overlays' / target_name)
        _reference(rgb, red, mask & ~red, decision, status).save(
            stage / 'doctor_foreground_reference_overlays' / target_name)
        _reference(rgb, red, mask & ~red, decision, status, audit=True).save(
            stage / 'ground_truth_audit_overlays' / target_name)
        ground_truth_mask, binary_mask = '', ''
        if status == 'completed_review_reference':
            corrected = np.zeros((*mask.shape, 4), dtype=np.uint8)
            corrected[mask] = (0, 255, 255, 178)
            ground_truth_mask = 'ground_truth/' + target_name
            binary_mask = 'ground_truth_binary/' + target_name
            Image.fromarray(corrected).save(stage / ground_truth_mask)
            Image.fromarray(mask.astype(np.uint8) * 255).save(stage / binary_mask)
        rows.append(dict(rank=rank, filename=name,
                         patient_id=reference['patient_id'] if reference else '',
                         split=reference['split'] if reference else 'hold',
                         decision=decision, reference_status=status,
                         doctor_reference='doctor_foreground_reference_overlays/' + target_name,
                         ground_truth_mask=ground_truth_mask, binary_mask=binary_mask,
                         reconstruction_mode=mode, image_sha256=image_hash,
                         segmentation_sha256=record['segmentation_sha256'],
                         annotation_sha256=record.get('annotation_sha256') or '',
                         hold_reason='' if reference else 'not_in_verified_patient_assignment'))
    write_csv(stage / 'manifest.csv', rows, list(rows[0]))
    write_csv(stage / 'ground_truth_status.csv', rows, list(rows[0]))
    active = [r for r in rows if r['split'] in ('train', 'val')]
    train_ids = {r['patient_id'] for r in active if r['split'] == 'train'}
    val_ids = {r['patient_id'] for r in active if r['split'] == 'val'}
    if train_ids & val_ids or not active:
        raise ValueError('Patient split is overlapping or empty')
    project_dir = stage / 'practice_project'
    for folder in ('annotations/train', 'annotations/val', 'segmentations', 'models',
                   'messages', 'logs', 'reviews', 'review_exports'):
        (project_dir / folder).mkdir(parents=True)
    info = dict(name=output.name, dataset=Path(os.path.relpath(output / 'images', syncdir / 'datasets')).as_posix(),
                location=Path(os.path.relpath(output / 'practice_project', syncdir)).as_posix(),
                original_model_file='random weights', file_names=[r['filename'] for r in active],
                train_file_names=[r['filename'] for r in active if r['split'] == 'train'],
                val_file_names=[r['filename'] for r in active if r['split'] == 'val'],
                split_method='preserved_verified_patient_assignments_v1',
                image_order_method='source_review_order', image_order_seed=0, training_seed=0,
                model_type=model_type, review_schema_version=1, auto_start_training_after_reviews=10)
    (project_dir / (output.name + '.seg_proj')).write_text(json.dumps(info, indent=2), encoding='utf-8')
    for source, filename in ((project / 'reviews/review_records.json', 'source_review_records.json'),
                             (next(project.glob('*.seg_proj')), 'source_project.seg_proj'),
                             (replay / 'manifest.csv', 'patient_assignment_source.csv')):
        shutil.copyfile(source, stage / 'provenance' / filename)
    summary = dict(images=len(rows), decisions=dict(Counter(r['decision'] for r in rows)),
                   reference_pictures=len(rows), usable_review_masks=sum(bool(r['binary_mask']) for r in rows),
                   practice_images=len(active), held_images=len(rows) - len(active),
                   train_images=len(info['train_file_names']), val_images=len(info['val_file_names']),
                   patient_overlap=0, blank_project=True, model_type=model_type,
                   final_test_created=False)
    readme = '''FRESH ANNOTATION PRACTICE - RAPHAEL COMPLETED REVIEWS

START HERE
In RetinaPainter use Project > Open project and select the .seg_proj file in
practice_project/. This starts a fresh U-Net project with blank annotations and
no trained model, using the preserved patient split. Keep the existing sync
directory. Do not recreate the project through New Project: this checkout's
New Project dialog only assigns individual files and ignores patient IDs.

1. images/: clean original scans. These are the images you annotate.
2. doctor_foreground_reference_overlays/: matching pictures with the doctor's
   decision and red RIPL marks. Open beside RetinaPainter as your answer guide.
   The scan filename matches its reference picture. Nothing is online.

doctor_foreground_overlays/ has the same red overlay without a footer.
doctor_foreground_masks/ contains the doctor's sparse red marks alone.
ground_truth_audit_overlays/ also shows accepted model foreground in cyan.
ground_truth/ and ground_truth_binary/ contain usable reconstructed definite
review masks. These are clinician-reviewed references, not independently drawn
exhaustive contours. The reference answers are never preloaded into your new
project's annotations or decisions.

All completed reviews have a reference picture, including UNCERTAIN cases.
Uncertain reviews do not have a definitive mask. A PRESENT review with an empty
recorded mask is explicitly flagged for doctor clarification, not made negative.
ground_truth_status.csv identifies each status and available file.

Images without verified patient routing remain in images/ and the references but
are withheld from the prepared project's navigation. Their manifest split is
hold; do not assign them to training or validation by guessing their identity.
This is a development/practice run, not an untouched final test dataset.

provenance/ preserves the doctor's original review records and correction
snapshots for audit. Do not copy them into the fresh project's annotations.
No prior training corrections, current decisions, or trained models are loaded.
All clinical files must stay local and out of assistant/chat/cloud services.

AGGREGATE COUNTS
'''
    (stage / 'README.txt').write_text(readme + json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if any(sha256(path) != digest for path, digest in watched.items()) or _fingerprints(replay) != frozen:
        raise ValueError('Sources changed during export')
    summary['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata = {**summary, 'file_sha256': _fingerprints(stage)}
    (stage / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if output.exists():
        raise ValueError('Output appeared while exporting')
    stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('project', 'replay', 'syncdir', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = build_practice_dataset(args.project, args.replay, args.syncdir, args.output)
    except Exception as error:
        parser.exit(2, 'Practice dataset was not published (' + type(error).__name__ + ').\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
