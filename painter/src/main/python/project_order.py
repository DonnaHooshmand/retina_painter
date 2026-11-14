"""Helpers for reproducible image order and train/validation splits.

Medical image sets usually hold several images per patient. A model that has
trained on one image of a patient is not being tested when it is validated on
another, so the split can be made by patient: ``patient_ids_from_filename_part``
or ``patient_ids_from_csv`` say which patient each image belongs to, and
``patient_grouped_train_val_split`` keeps every patient on one side.
"""

import csv
import os
import random
import re
import statistics
from collections import Counter
from pathlib import Path

# Characters that commonly separate the parts of a clinical export file name.
DEFAULT_SEPARATORS = '_- .'


def seeded_file_order(fnames, seed):
    """Return a reproducibly shuffled copy of ``fnames``.

    Sorting before shuffling makes the result independent of filesystem list
    order. A local RNG avoids changing randomness used elsewhere in the
    painter (for example, palette generation).
    """
    ordered_fnames = sorted(fnames)
    random.Random(seed).shuffle(ordered_fnames)
    return ordered_fnames


def manifest_file_order(dataset_dir, fnames):
    """Return a manifest-ranked order when the dataset has a manifest.

    A manifest may live in the selected dataset directory or its parent. The
    latter supports numeric nested folders such as ``150`` and ``250`` that
    share one root manifest. If a recognizable manifest is present, every
    selected image must occur in it; silently falling back to a different
    order would invalidate nested-prefix experiments.

    Returns ``(None, None)`` when no manifest exists, otherwise the ordered
    filenames and the resolved manifest path.
    """
    dataset_dir = Path(dataset_dir)
    fnames = list(fnames)
    candidates = (
        dataset_dir / "manifest.csv",
        dataset_dir.parent / "manifest.csv",
    )
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return None, None

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"rank", "filename"}.issubset(
                reader.fieldnames):
            raise ValueError(
                f"Dataset manifest must contain rank and filename columns: "
                f"{manifest_path}")
        ranked_names = []
        seen_ranks = set()
        seen_names = set()
        for row in reader:
            try:
                rank = int(row["rank"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid rank in dataset manifest {manifest_path}: "
                    f"{row.get('rank')}") from error
            name = row["filename"]
            name_key = name.casefold()
            if rank < 1 or rank in seen_ranks:
                raise ValueError(
                    f"Duplicate or invalid rank {rank} in {manifest_path}")
            if not name or name_key in seen_names:
                raise ValueError(
                    f"Duplicate or empty filename {name!r} in {manifest_path}")
            seen_ranks.add(rank)
            seen_names.add(name_key)
            ranked_names.append((rank, name_key))

    selected_by_key = {}
    for name in fnames:
        key = name.casefold()
        if key in selected_by_key:
            raise ValueError(f"Duplicate dataset filename: {name}")
        selected_by_key[key] = name
    manifest_keys = {name_key for _, name_key in ranked_names}
    missing = sorted(
        (selected_by_key[key] for key in set(selected_by_key) - manifest_keys),
        key=str.casefold)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Dataset image(s) missing from manifest {manifest_path}: {preview}")

    ordered = [
        selected_by_key[name_key]
        for _, name_key in sorted(ranked_names)
        if name_key in selected_by_key
    ]
    if len(ordered) != len(fnames):
        raise ValueError(
            f"Manifest ordering did not cover every dataset image in "
            f"{dataset_dir}")
    return ordered, manifest_path


def fixed_train_val_split(ordered_fnames, train_per_val=5):
    """Assign an already ordered image list to a fixed train/val split.

    The first image and then every ``train_per_val + 1`` image is assigned to
    validation. This matches the painter's inherited 5:1 count-based routing
    pattern, but fixes the assignment before annotation starts. Blank or
    delayed annotations therefore cannot change which split a filename belongs
    to, and matched projects with the same image-order seed get the same split.
    """
    if train_per_val < 1:
        raise ValueError("train_per_val must be at least 1")

    cycle = train_per_val + 1
    val_fnames = [fname for index, fname in enumerate(ordered_fnames)
                  if index % cycle == 0]
    train_fnames = [fname for index, fname in enumerate(ordered_fnames)
                    if index % cycle != 0]
    return train_fnames, val_fnames


def filename_parts(fname, separators=DEFAULT_SEPARATORS):
    """Split a file name, without its extension, at the separator characters.

    ``'A17_OD_2024-03-01_012.png'`` becomes ``['A17', 'OD', '2024', '03',
    '01', '012']`` with the default separators. Runs of separators count as
    one, and an empty ``separators`` returns the whole name as a single part.
    """
    stem = os.path.splitext(fname)[0]
    if not separators:
        return [stem]
    pattern = '[' + re.escape(separators) + ']+'
    return [part for part in re.split(pattern, stem) if part]


def patient_ids_from_filename_part(fnames, part,
                                   separators=DEFAULT_SEPARATORS):
    """Read each image's patient from one part of its file name.

    Returns ``(patient_of, unmatched)``: a ``{file name: patient}`` mapping
    and the file names that do not have that many parts. Patients are compared
    without regard to letter case.
    """
    patient_of = {}
    unmatched = []
    for fname in fnames:
        parts = filename_parts(fname, separators)
        if 0 <= part < len(parts):
            patient_of[fname] = parts[part].casefold()
        else:
            unmatched.append(fname)
    return patient_of, unmatched


def suggest_patient_part(fnames, separators=DEFAULT_SEPARATORS):
    """Guess which part of the file names identifies the patient.

    Returns the index of the first part, present in every name, that is
    neither the same for all images nor different for every image, or ``None``
    when no part groups the images. This is only a starting point for the
    user, who sees the resulting patient count before creating a project.
    """
    split_names = [filename_parts(fname, separators) for fname in fnames]
    if not split_names:
        return None
    for index in range(min(len(parts) for parts in split_names)):
        distinct = len({parts[index].casefold() for parts in split_names})
        if 1 < distinct < len(split_names):
            return index
    return None


def patient_ids_from_csv(csv_path, fnames):
    """Read each image's patient from a CSV with a patient for every image.

    The file needs ``filename`` and ``patient_id`` columns; other columns are
    ignored and file names are matched without regard to letter case. Every
    image in *fnames* must appear with a non-empty patient, because a silently
    ungrouped image could put one patient on both sides of the split.
    """
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {'filename', 'patient_id'}.issubset(
                reader.fieldnames):
            raise ValueError(
                'The patient list must have filename and patient_id columns.')
        listed = {}
        for row in reader:
            name = (row['filename'] or '').strip().casefold()
            patient = (row['patient_id'] or '').strip().casefold()
            if not name:
                continue
            if listed.get(name, patient) != patient:
                raise ValueError(
                    'The patient list gives one image two different patients.')
            listed[name] = patient
    patient_of = {}
    missing = 0
    for fname in fnames:
        patient = listed.get(fname.casefold())
        if patient:
            patient_of[fname] = patient
        else:
            missing += 1
    if missing:
        raise ValueError(
            f'{missing} of {len(fnames)} images have no patient in the '
            'patient list.')
    return patient_of


def grouping_summary(fnames, patient_of):
    """Describe a patient grouping with counts only (no identifiers)."""
    sizes = sorted(Counter(
        patient_of[fname] for fname in fnames if fname in patient_of).values())
    return {
        'images': len(fnames),
        'ungrouped_images': sum(fname not in patient_of for fname in fnames),
        'patients': len(sizes),
        'median_images_per_patient': (
            statistics.median(sizes) if sizes else 0),
        'largest_patient_images': sizes[-1] if sizes else 0,
    }


def patient_grouped_train_val_split(ordered_fnames, patient_of,
                                    train_per_val=5):
    """Assign whole patients to train or validation along the image order.

    Images are visited in navigation order. The first time a patient appears,
    the patient goes to validation if validation holds less than its
    ``1 : train_per_val`` share of the images seen so far, and to training
    otherwise; every image of that patient then follows. This has three
    useful properties:

    * no patient is ever on both sides;
    * about one image in ``train_per_val + 1`` is validation at every point
      of an annotation session, not just at the end, so validation data
      arrives as early as it does with the per-image rule;
    * a patient's side depends only on the images before its first one, so a
      project built from a prefix of the same order (nested datasets) gives
      every shared patient the same side.

    When every image is its own patient the result equals
    ``fixed_train_val_split``.
    """
    if train_per_val < 1:
        raise ValueError("train_per_val must be at least 1")
    missing = [fname for fname in ordered_fnames if fname not in patient_of]
    if missing:
        raise ValueError(
            f'{len(missing)} image(s) have no patient; cannot split by '
            'patient.')

    side_of = {}
    seen_train = seen_val = 0
    train_fnames, val_fnames = [], []
    for fname in ordered_fnames:
        patient = patient_of[fname]
        if patient not in side_of:
            validation_is_due = seen_val * train_per_val <= seen_train
            side_of[patient] = 'val' if validation_is_due else 'train'
        if side_of[patient] == 'val':
            seen_val += 1
            val_fnames.append(fname)
        else:
            seen_train += 1
            train_fnames.append(fname)
    return train_fnames, val_fnames


def patient_group_numbers(ordered_fnames, patient_of):
    """Number patients by first appearance; one number per image, in order.

    Project files store these numbers rather than the identifiers themselves,
    so the grouping can be audited without copying patient IDs around.
    """
    number_of = {}
    return [number_of.setdefault(patient_of[fname], len(number_of) + 1)
            for fname in ordered_fnames]
