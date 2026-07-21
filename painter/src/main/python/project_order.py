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
