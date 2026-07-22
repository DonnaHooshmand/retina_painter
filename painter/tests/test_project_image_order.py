"""Tests for reproducible image ordering across matched projects."""

import os
import random
import sys


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from project_order import fixed_train_val_split, seeded_file_order


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


def test_fixed_split_matches_five_train_to_one_validation():
    ordered = [f'scan_{i:02d}.png' for i in range(18)]

    train, val = fixed_train_val_split(ordered)

    assert val == [ordered[0], ordered[6], ordered[12]]
    assert len(train) == 15
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(ordered)


def test_seed_fixes_both_navigation_order_and_split():
    fnames = [f'scan_{i:02d}.png' for i in range(30)]

    first_order = seeded_file_order(fnames, seed=42)
    second_order = seeded_file_order(reversed(fnames), seed=42)

    assert first_order == second_order
    assert fixed_train_val_split(first_order) == fixed_train_val_split(
        second_order)
