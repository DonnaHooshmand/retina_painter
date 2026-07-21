"""Tests for reproducible image ordering across matched projects."""

import os
import random
import sys


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from project_order import seeded_file_order


def test_same_seed_and_files_produce_same_order():
    fnames = [f'scan_{i:02d}.png' for i in range(20)]

    first = seeded_file_order(fnames, seed=42)
    second = seeded_file_order(reversed(fnames), seed=42)

    assert first == second
    assert sorted(first) == sorted(fnames)


def test_different_seed_changes_order():
    fnames = [f'scan_{i:02d}.png' for i in range(20)]

    assert seeded_file_order(fnames, seed=1) != seeded_file_order(
        fnames, seed=2)


def test_seeded_order_does_not_mutate_input_or_global_rng():
    fnames = ['c.png', 'a.png', 'b.png']
    original = list(fnames)
    random.seed(123)
    state_before = random.getstate()

    seeded_file_order(fnames, seed=0)

    assert fnames == original
    assert random.getstate() == state_before
