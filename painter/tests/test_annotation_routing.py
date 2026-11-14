"""Tests for fixed seeded annotation routing."""

import os
import sys

import pytest


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

import numpy as np
from PIL import Image

import file_utils
from file_utils import (annotation_is_empty, remove_annotation,
                        resolve_new_annot_target_dir)


def test_empty_annotation_is_recognised_and_removed(tmp_path, monkeypatch):
    strokes = np.zeros((5, 6, 4), dtype=np.uint8)
    empty = tmp_path / 'empty.png'
    Image.fromarray(strokes, mode='RGBA').save(empty)
    strokes[2, 3] = [0, 255, 0, 180]
    painted = tmp_path / 'painted.png'
    Image.fromarray(strokes, mode='RGBA').save(painted)

    assert annotation_is_empty(empty)
    assert not annotation_is_empty(painted)

    # The trainer may have the file open for a moment (Windows): the delete
    # is retried, and a file that stays locked is reported as not removed.
    real_remove = file_utils.os.remove
    refusals = [PermissionError('in use'), PermissionError('in use')]

    def remove(path):
        if refusals:
            raise refusals.pop()
        real_remove(path)

    monkeypatch.setattr(file_utils.os, 'remove', remove)
    assert remove_annotation(empty, delay=0)
    assert not empty.exists()
    assert remove_annotation(empty, delay=0)  # already gone

    refusals.extend([PermissionError('in use')] * 3)
    assert not remove_annotation(painted, attempts=3, delay=0)
    assert painted.exists()


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
