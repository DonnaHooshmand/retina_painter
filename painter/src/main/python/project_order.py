"""Helpers for reproducible image order in painter projects."""

import random


def seeded_file_order(fnames, seed):
    """Return a reproducibly shuffled copy of ``fnames``.

    Sorting before shuffling makes the result independent of filesystem list
    order. A local RNG avoids changing randomness used elsewhere in the
    painter (for example, palette generation).
    """
    ordered_fnames = sorted(fnames)
    random.Random(seed).shuffle(ordered_fnames)
    return ordered_fnames


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
