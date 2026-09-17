"""Helpers for reproducible image order in painter projects."""

import csv
import random
from pathlib import Path


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
