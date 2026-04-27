"""
Tests for the on-disk annotation schema (Phase 2).

The painter writes RGBA PNGs. By convention:

  * channel 0 (R) = foreground brush
  * channel 1 (G) = background brush
  * channel 2 (B) = unsure brush  (Phase 2 — legacy projects have B=0)
  * channel 3 (A) = pixel-painted alpha

These tests verify the dataset reads that schema correctly:

  1. A 3-channel annotation produces ``mask = (fg | bg) & ~unsure``
     and a separate ``unsure`` tensor.
  2. A legacy 2-channel-style annotation (B=0 everywhere) reads
     identically to before — backward compatibility holds.
  3. Overlapping fg/bg/unsure pixels trigger an assertion (corruption
     detection).
"""

import os
import random
import sys
import tempfile

import numpy as np
import pytest
import torch
from skimage.io import imsave

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

# pylint: disable=C0413
from datasets import TrainDataset


def _write_synthetic_project(tmpdir, fg_mask, bg_mask, unsure_mask):
    """Write a synthetic image + annotation PNG pair into tmpdir.

    Returns (dataset_dir, train_annot_dir) ready to feed TrainDataset.
    """
    h, w = fg_mask.shape

    dataset_dir = os.path.join(tmpdir, 'images')
    annot_dir = os.path.join(tmpdir, 'annotations', 'train')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    # Synthetic RGB image (random gradient).
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    image[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    imsave(os.path.join(dataset_dir, 'sample.png'), image, check_contrast=False)

    # Annotation RGBA: R=fg, G=bg, B=unsure, A=255 wherever any of those
    # are set (alpha is unused at the trainer but we set it anyway to
    # mimic what the painter writes).
    annot = np.zeros((h, w, 4), dtype=np.uint8)
    annot[:, :, 0] = fg_mask.astype(np.uint8) * 255
    annot[:, :, 1] = bg_mask.astype(np.uint8) * 255
    annot[:, :, 2] = unsure_mask.astype(np.uint8) * 255
    any_painted = (fg_mask | bg_mask | unsure_mask).astype(np.uint8)
    annot[:, :, 3] = any_painted * 255
    imsave(os.path.join(annot_dir, 'sample.png'), annot, check_contrast=False)

    return dataset_dir, annot_dir


class _NoOpAugmentor:
    """Stand-in for ``UNetTransformer`` that performs no augmentation.

    The real augmentor applies elastic warps, color jitter, flips, etc.
    Those are the right behavior for training, but they make
    pixel-level schema assertions flaky. Schema tests only care about
    the read + mask logic, so we bypass augmentation here.
    """
    def transform(self, photo, annot):
        return photo, annot


def _pull_tile(dataset_dir, annot_dir, in_w=64, out_w=64, seed=0):
    """Build TrainDataset and pull one tile, returning the tuple of tensors.

    Augmentation is disabled so the returned tensors reflect the on-disk
    annotation directly. The random crop is seeded explicitly so the
    tests are deterministic regardless of test ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    ds = TrainDataset(annot_dir, dataset_dir, in_w=in_w, out_w=out_w,
                      min_epoch_tiles=1)
    ds.augmentor = _NoOpAugmentor()
    return ds[0]


# ---------------------------------------------------------------------------
# 1. Three-channel annotation round-trip
# ---------------------------------------------------------------------------
#
# These tests fully cover the image with fg/bg/unsure (no untouched
# pixels) so the result is invariant to where the random crop lands —
# even crops that pull from reflected padding still see fg+bg+unsure.
# This isolates the schema/mask logic from the (intentional) randomness
# of tile selection.

def test_three_channel_annotation_returns_four_tuple():
    """The dataset must return a 4-tuple (image, fg, mask, unsure) at the
    expected shapes, regardless of crop position."""
    h = w = 128
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.zeros((h, w), dtype=bool)

    # Tile evenly with all three brushes so any crop has a representative
    # mix. Use a small block-stripe pattern so reflected padding also
    # contains all three types.
    block = 16
    for col in range(0, w, block * 3):
        fg_mask[:, col:col + block] = True
        bg_mask[:, col + block:col + 2 * block] = True
        unsure_mask[:, col + 2 * block:col + 3 * block] = True

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        result = _pull_tile(dd, ad, in_w=h, out_w=h)

    assert len(result) == 4, \
        f"Expected (image, fg, mask, unsure); got {len(result)}-tuple"
    image, foreground, mask, unsure = result

    assert image.shape == (3, h, w)
    assert foreground.shape == (h, w)
    assert mask.shape == (h, w)
    assert unsure.shape == (h, w)

    # Sanity: at least some pixels of each type should reach the tile
    # given the block-stripe coverage.
    assert (mask.numpy() > 0).sum() > 0, "mask tensor should not be empty"
    assert (unsure.numpy() > 0).sum() > 0, "unsure tensor should not be empty"


def test_mask_and_unsure_are_disjoint():
    """The critical Phase-2 invariant: mask and unsure tensors must
    never overlap. A pixel marked unsure must be excluded from the
    supervised mask, regardless of crop position or annotation
    geometry.
    """
    h = w = 128
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.zeros((h, w), dtype=bool)

    block = 16
    for col in range(0, w, block * 3):
        fg_mask[:, col:col + block] = True
        bg_mask[:, col + block:col + 2 * block] = True
        unsure_mask[:, col + 2 * block:col + 3 * block] = True

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        # Run several seeds — the disjoint invariant must hold for every
        # crop, not just one lucky position.
        for seed in range(5):
            _, _, mask, unsure = _pull_tile(dd, ad, in_w=h, out_w=h, seed=seed)
            overlap = (mask.numpy() > 0) & (unsure.numpy() > 0)
            assert not overlap.any(), (
                f"seed={seed}: mask and unsure overlap at "
                f"{overlap.sum()} pixels — broken contract"
            )


# ---------------------------------------------------------------------------
# 2. Backward compatibility with legacy 2-channel-style annotations
# ---------------------------------------------------------------------------

def test_legacy_two_channel_annotation_reads_with_zero_unsure():
    """A PNG with channel 2 = 0 everywhere (legacy projects) must read
    with unsure all zero and behaviour identical to pre-Phase-2.
    """
    h = w = 128
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.zeros((h, w), dtype=bool)  # legacy: always 0

    # Cover the whole image so ``mask`` should be all-1s regardless of
    # crop position (reflection of fg+bg is still fg+bg).
    fg_mask[:, :64] = True
    bg_mask[:, 64:] = True

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        result = _pull_tile(dd, ad, in_w=h, out_w=h)

    image, foreground, mask, unsure = result

    # Unsure tensor must be all zero for legacy projects.
    assert torch.all(unsure == 0), \
        "legacy 2-channel annotation produced non-zero unsure tensor"

    # mask should be 1 essentially everywhere (full coverage). Allow a
    # 1-pixel slack to absorb any boundary edge case.
    mask_count = (mask.numpy() > 0).sum()
    assert mask_count >= h * w - 1, (
        f"legacy fully-covered annotation should have mask≈full; "
        f"got {mask_count} of {h*w}"
    )


# ---------------------------------------------------------------------------
# 3. Overlap assertion fires on corrupted annotations
# ---------------------------------------------------------------------------

def test_overlapping_fg_bg_pixels_raise():
    """A pixel marked as both fg AND bg should trigger the assertion."""
    h = w = 64
    fg_mask = np.ones((h, w), dtype=bool)
    bg_mask = np.ones((h, w), dtype=bool)   # full overlap
    unsure_mask = np.zeros((h, w), dtype=bool)

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        with pytest.raises(AssertionError, match="overlapping fg/bg"):
            _pull_tile(dd, ad, in_w=h, out_w=h)


def test_overlapping_fg_unsure_pixels_raise():
    """A pixel marked as both fg AND unsure should trigger the assertion."""
    h = w = 64
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.zeros((h, w), dtype=bool)

    fg_mask[:32, :] = True
    unsure_mask[:32, :] = True   # overlaps the fg region
    bg_mask[32:, :] = True       # bg in the other half so the tile isn't empty

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        with pytest.raises(AssertionError, match="overlapping fg/unsure"):
            _pull_tile(dd, ad, in_w=h, out_w=h)


def test_overlapping_bg_unsure_pixels_raise():
    """A pixel marked as both bg AND unsure should trigger the assertion."""
    h = w = 64
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.zeros((h, w), dtype=bool)

    bg_mask[:32, :] = True
    unsure_mask[:32, :] = True   # overlaps the bg region
    fg_mask[32:, :] = True       # fg in the other half so the tile isn't empty

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        with pytest.raises(AssertionError, match="overlapping bg/unsure"):
            _pull_tile(dd, ad, in_w=h, out_w=h)


# ---------------------------------------------------------------------------
# 4. Unsure-only tiles are skipped by the tile-selection loop
# ---------------------------------------------------------------------------

def test_unsure_only_tiles_are_skipped():
    """If most of the image is unsure with only a small fg+bg region,
    the tile-selection loop must keep looking until it finds a crop
    with fg+bg pixels. Otherwise the model would waste a forward+
    backward pass on a fully-masked-out tile.

    We sample multiple random crops; every returned tile must have
    mask > 0. (The legacy ``np.sum(annot_tile) > 0`` check would have
    accepted unsure-only tiles since unsure is in channel 2; the new
    check ``np.sum(annot_tile[:, :, :2]) > 0`` excludes that.)
    """
    h = w = 128
    fg_mask = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)
    unsure_mask = np.ones((h, w), dtype=bool)

    # Small fg+bg patch; rest is unsure.
    fg_mask[40:80, 40:60] = True
    bg_mask[40:80, 60:80] = True
    unsure_mask[fg_mask] = False
    unsure_mask[bg_mask] = False

    with tempfile.TemporaryDirectory() as tmp:
        dd, ad = _write_synthetic_project(tmp, fg_mask, bg_mask, unsure_mask)
        ds = TrainDataset(ad, dd, in_w=h, out_w=h, min_epoch_tiles=1)
        ds.augmentor = _NoOpAugmentor()

        # Pull 10 tiles with different seeds. Every one must have
        # mask > 0 — i.e. the loop never accepted an unsure-only tile.
        for seed in range(10):
            random.seed(seed)
            np.random.seed(seed)
            _, _, mask, _ = ds[0]
            assert mask.sum().item() > 0, (
                f"seed={seed}: tile-selection returned fully-masked-out "
                f"tile despite fg+bg pixels existing in source"
            )
