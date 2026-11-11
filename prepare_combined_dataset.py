"""Combine all reviewed curated datasets into one unified development dataset.

Sources:
  1. Raphael_Updated_20261021_v1  — 316 images, merged original + follow-up reviews
  2. Raphael_Positive_Curated_20261105_v1 — 200 images, positive-candidate session

Produces a single dataset in the standard ground-truth format with no image
overlap and no patient-level split conflicts.  No network access, model
training, or changes to source directories.  CLI output is aggregate only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / 'painter/src/main/python'))
from review_utils import reconstruct_review_binary

from prepare_curated_training_dataset import sha256, write_csv


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path, format='PNG')


def _read_rgba(path: Path, expected_wh):
    with Image.open(path) as img:
        img.load()
        if img.size != expected_wh:
            raise ValueError(f'Image size {img.size} != expected {expected_wh} for {path.name}')
        return np.asarray(img.convert('RGBA'))


def _annotation_pixels(rgba: np.ndarray):
    red = (rgba[:, :, 0] > 0) & (rgba[:, :, 3] > 0)
    green = (rgba[:, :, 1] > 0) & (rgba[:, :, 3] > 0)
    return red, green


def _copy_verified(src: Path, dst: Path, expected_sha: str | None = None) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    actual = sha256(dst)
    if expected_sha and actual != expected_sha:
        raise ValueError(f'Hash mismatch copying {src.name}')
    return actual


# ---------------------------------------------------------------------------
# Source 1: Raphael_Updated_20261021_v1
# ---------------------------------------------------------------------------

def _load_updated(ds: Path, ripl_assignments: dict[str, str]):
    """Return list of dicts with one entry per image."""
    with open(ds / 'manifest.csv', encoding='utf-8-sig', newline='') as f:
        manifest_rows = {r['filename']: r for r in csv.DictReader(f)}

    # Provenance annotation snapshots — followup supersedes original
    orig_snaps = {
        p.stem: p
        for p in (ds / 'provenance/original/reviews/annotation_snapshots').glob('*.png')
    }
    fu_snaps = {
        p.stem: p
        for p in (ds / 'provenance/followup/source_snapshot/reviews/annotation_snapshots').glob('*.png')
    }
    # merge: followup wins
    annot_snaps = {**orig_snaps, **fu_snaps}

    # Reviewed masks
    reviewed_masks = {p.stem: p for p in (ds / 'ground_truth').glob('*.png')}

    records = []
    for filename, row in manifest_rows.items():
        pid = (row.get('patient_id') or '').strip()
        decision = row.get('decision', '')

        # Resolve split
        split = row.get('prior_development_split', '').strip()
        if not split:
            split = ripl_assignments.get(pid, 'train')  # 2 unknown patients → train

        image_path = ds / 'images' / filename
        if not image_path.exists():
            raise ValueError(f'Missing image: {image_path}')

        stem = Path(filename).stem
        annot_path = annot_snaps.get(stem)
        reviewed_path = reviewed_masks.get(stem)

        records.append(dict(
            filename=filename,
            patient_id=pid or 'unknown',
            split=split,
            decision=decision,
            image_path=image_path,
            image_sha=row.get('image_sha256', ''),
            annot_path=annot_path,
            reviewed_path=reviewed_path,
            source='Raphael_Updated_20261021_v1',
        ))
    return records


# ---------------------------------------------------------------------------
# Source 2: Raphael_Positive_Curated_20261105_v1
# ---------------------------------------------------------------------------

def _load_positive_curated(ds: Path):
    """Return list of dicts with one entry per image."""
    with open(ds / 'ground_truth/bscan_labels.csv', encoding='utf-8-sig', newline='') as f:
        label_rows = {r['filename']: r for r in csv.DictReader(f)}

    with open(ds / 'ground_truth/sparse_mask_index.csv', encoding='utf-8-sig', newline='') as f:
        sparse_rows = {r['filename']: r for r in csv.DictReader(f)}

    with open(ds / 'ground_truth/reviewed_mask_index.csv', encoding='utf-8-sig', newline='') as f:
        reviewed_rows = {r['filename']: r for r in csv.DictReader(f)}

    records = []
    for filename, lrow in label_rows.items():
        split = lrow['split']
        stem = Path(filename).stem

        # Annotation from annotations/split/stem.png
        annot_path = ds / 'annotations' / split / (stem + '.png')
        if not annot_path.exists():
            annot_path = None

        # Reviewed mask
        rm = reviewed_rows.get(filename, {})
        reviewed_path = ds / 'ground_truth' / rm['mask'] if rm.get('mask') else None

        image_path = ds / 'images' / filename
        if not image_path.exists():
            raise ValueError(f'Missing image: {image_path}')

        records.append(dict(
            filename=filename,
            patient_id=lrow['patient_id'],
            split=split,
            decision=lrow['decision'],
            image_path=image_path,
            image_sha='',
            annot_path=annot_path,
            reviewed_path=reviewed_path,
            source='Raphael_Positive_Curated_20261105_v1',
        ))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_combined_dataset(updated_dir, positive_dir, output_dir):
    updated = Path(updated_dir).resolve()
    positive = Path(positive_dir).resolve()
    output = Path(output_dir).resolve()

    if output.exists():
        raise ValueError('Output directory already exists; refusing to overwrite')

    # Load RIPL_Dev patient assignments for split fallback
    ripl_pa_path = updated.parent.parent / 'datasets/curated/_datasets/RIPL_Development_20260925_v1/patient_assignments.csv'
    ripl_assignments: dict[str, str] = {}
    if ripl_pa_path.exists():
        with open(ripl_pa_path, encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                ripl_assignments[row['patient_id']] = row['split']

    records_upd = _load_updated(updated, ripl_assignments)
    records_pos = _load_positive_curated(positive)

    # Verify no image content overlap
    upd_names = {r['filename'] for r in records_upd}
    pos_names = {r['filename'] for r in records_pos}
    overlap = upd_names & pos_names
    if overlap:
        raise ValueError(f'{len(overlap)} duplicate filename(s) between sources: {sorted(overlap)[:3]}')

    all_records = records_upd + records_pos

    # Verify patient-level split consistency
    patient_splits: dict[str, str] = {}
    for r in all_records:
        pid = r['patient_id']
        if pid == 'unknown':
            continue
        if pid in patient_splits and patient_splits[pid] != r['split']:
            raise ValueError(f'Patient {pid} has conflicting splits: '
                             f'{patient_splits[pid]} vs {r["split"]}')
        patient_splits[pid] = r['split']

    # Atomic staging
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.' + output.name + '.partial-' + uuid4().hex)
    stage.mkdir()

    gt = stage / 'ground_truth'
    for folder in ('clinician_foreground', 'known_pixels', 'reviewed_masks'):
        (gt / folder).mkdir(parents=True)

    labels, sparse_index, reviewed_index = [], [], []
    image_count = 0
    eligible_count = 0
    reviewed_count = 0

    for rec in all_records:
        filename = rec['filename']
        split = rec['split']
        decision = rec['decision']
        pid = rec['patient_id']
        stem = Path(filename).stem

        # Copy image
        dest_img = stage / 'images' / filename
        actual_sha = _copy_verified(rec['image_path'], dest_img,
                                    rec['image_sha'] or None)
        image_count += 1

        # Determine expected image size
        with Image.open(dest_img) as img:
            wh = img.size

        definite = decision in ('present', 'absent')

        labels.append(dict(
            filename=filename,
            patient_id=pid,
            split=split,
            decision=decision,
            binary_label=int(decision == 'present') if definite else '',
            binary_evaluation_eligible=definite,
            source=rec['source'],
            independently_adjudicated=False,
        ))

        # Sparse annotation
        eligible = False
        red_pixels = 0
        green_pixels = 0
        annot_sha = ''
        exclusion_reason = ''

        if decision == 'uncertain':
            exclusion_reason = 'uncertain_decision'
        elif rec['annot_path'] is None or not rec['annot_path'].exists():
            exclusion_reason = 'no_annotation_file'
        else:
            rgba = _read_rgba(rec['annot_path'], wh)
            red, green = _annotation_pixels(rgba)
            if not (red | green).any():
                exclusion_reason = 'annotation_has_no_pixels'
            elif decision == 'absent' and red.any():
                exclusion_reason = 'absent_annotation_has_red_pixels'
            else:
                eligible = True
                red_pixels = int(red.sum())
                green_pixels = int(green.sum())
                annot_sha = sha256(rec['annot_path'])

                # Copy annotation
                annot_name = stem + '.png'
                dest_annot = stage / 'annotations' / split / annot_name
                _copy_verified(rec['annot_path'], dest_annot)
                _png(red, gt / 'clinician_foreground' / annot_name)
                _png(red | green, gt / 'known_pixels' / annot_name)
                eligible_count += 1

        sparse_index.append(dict(
            filename=filename,
            split=split,
            decision=decision,
            source=rec['source'],
            training_eligible=eligible,
            foreground_mask='clinician_foreground/' + stem + '.png' if eligible else '',
            known_pixel_mask='known_pixels/' + stem + '.png' if eligible else '',
            red_pixels=red_pixels,
            green_pixels=green_pixels,
            annotation_sha256=annot_sha,
            status=exclusion_reason or 'available_sparse_only',
        ))

        # Reviewed mask
        reviewed = dict(
            filename=filename,
            split=split,
            decision=decision,
            source=rec['source'],
            mask='',
            status='no_reviewed_mask',
        )
        if definite and rec['reviewed_path'] and rec['reviewed_path'].exists():
            rel = 'reviewed_masks/' + stem + '.png'
            _copy_verified(rec['reviewed_path'], gt / rel)
            reviewed['mask'] = rel
            reviewed['status'] = 'available'
            reviewed_count += 1
        reviewed_index.append(reviewed)

    for csv_name, entries in (
            ('bscan_labels.csv', labels),
            ('sparse_mask_index.csv', sparse_index),
            ('reviewed_mask_index.csv', reviewed_index)):
        write_csv(gt / csv_name, entries, list(entries[0]))

    decisions = dict(Counter(r['decision'] for r in all_records))
    source_counts = dict(Counter(r['source'] for r in all_records))

    result = dict(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='combined_development_dataset',
        final_test_created=False,
        images=image_count,
        decisions=decisions,
        source_counts=source_counts,
        eligible_sparse_annotations=eligible_count,
        reviewed_masks=reviewed_count,
        splits={sp: dict(
            images=sum(r['split'] == sp for r in all_records),
            patients=len({r['patient_id'] for r in all_records if r['split'] == sp}))
            for sp in ('train', 'val')},
        sources=[str(updated), str(positive)],
    )

    readme = (
        'RIPL COMBINED DEVELOPMENT DATASET\n\n'
        'All reviewed data as of 2026-11-05. Two sources:\n'
        '  Raphael_Updated_20261021_v1  : original + follow-up reviews (316 images)\n'
        '  Raphael_Positive_Curated_20261105_v1 : positive-candidate session (200 images)\n\n'
        'Use images/ as the image directory in RetinaPainter.\n'
        'Patient splits are patient-separated with no conflicts across sources.\n\n'
        'ground_truth/bscan_labels.csv : scan-level decisions\n'
        'ground_truth/clinician_foreground/ : binary foreground masks (red pixels)\n'
        'ground_truth/known_pixels/ : binary known-pixel masks (red | green)\n'
        'ground_truth/reviewed_masks/ : reviewed reference masks\n'
        'ground_truth/sparse_mask_index.csv : annotation eligibility\n'
        'ground_truth/reviewed_mask_index.csv : reviewed mask availability\n\n'
        'This is a development dataset, not a final held-out test cohort.\n\n'
        'AGGREGATE SUMMARY\n'
    )
    (stage / 'COMBINED_DATASET_README.txt').write_text(
        readme + json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (stage / 'combined_dataset_metadata.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')

    if output.exists():
        raise ValueError('Output appeared during preparation; refusing replacement')
    stage.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updated', type=Path,
                        default='D:/RootPainterSync/curated_datasets/Raphael_Updated_20261021_v1',
                        help='Path to Raphael_Updated_20261021_v1')
    parser.add_argument('--positive', type=Path,
                        default='D:/RootPainterSync/curated_datasets/Raphael_Positive_Curated_20261105_v1',
                        help='Path to Raphael_Positive_Curated_20261105_v1')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output path for the combined dataset (must not exist)')
    args = parser.parse_args()
    try:
        result = prepare_combined_dataset(args.updated, args.positive, args.output)
    except Exception as error:
        parser.exit(2, 'Dataset was not published (' + type(error).__name__ + '). '
                    'Inspect sources locally.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
