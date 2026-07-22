"""Tests for fixed seeded annotation routing."""

import os
import sys

import pytest


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from file_utils import resolve_new_annot_target_dir


def test_fixed_target_overrides_count_router(tmp_path):
    train_dir = tmp_path / 'train'
    val_dir = tmp_path / 'val'
    train_dir.mkdir()
    val_dir.mkdir()

    assert resolve_new_annot_target_dir(
        train_dir, val_dir, fixed_target_dir=val_dir) == os.path.abspath(
            str(val_dir))
    assert resolve_new_annot_target_dir(
        train_dir, val_dir, fixed_target_dir=train_dir) == os.path.abspath(
            str(train_dir))


def test_fixed_target_rejects_directory_outside_split(tmp_path):
    train_dir = tmp_path / 'train'
    val_dir = tmp_path / 'val'
    other_dir = tmp_path / 'other'
    train_dir.mkdir()
    val_dir.mkdir()
    other_dir.mkdir()

    with pytest.raises(ValueError, match='not train or val'):
        resolve_new_annot_target_dir(
            train_dir, val_dir, fixed_target_dir=other_dir)
