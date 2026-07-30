"""
Copyright (C) 2019, 2020 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Annotation semantics
--------------------
The dataset returns three tensors per tile: ``image``, ``foreground`` (the
label, 1 where the clinician painted lesion, 0 elsewhere) and ``mask``
(1 where the pixel is supervised, 0 where it is untouched). Untouched
pixels must be ignored by the loss — see ``loss.py`` and
``docs/supervision_plan.md``.
"""

# pylint: disable=C0111, R0913, R0903, R0914, W0511
import random
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from PIL import Image
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

from im_utils import load_train_image_and_annot
from file_utils import ls
import im_utils
import elastic


def annotation_output_region(annot_tile, tile_pad):
    """Return the annotation pixels that the model can actually supervise."""
    if tile_pad > 0:
        return annot_tile[tile_pad:-tile_pad, tile_pad:-tile_pad]
    return annot_tile

def elastic_transform(photo, annot):
    def_map = elastic.get_elastic_map(photo.shape,
                                      scale=random.random(),
                                      intensity=0.4 + (0.6 * random.random()))
    photo = elastic.transform_image(photo, def_map)
    annot = elastic.transform_image(annot, def_map, channels=2)
    annot = np.round(annot).astype(np.int64)
    return photo, annot

def guassian_noise_transform(photo, annot):
    sigma = np.abs(np.random.normal(0, scale=0.09))
    photo = im_utils.add_gaussian_noise(photo, sigma)
    return photo, annot

def salt_pepper_transform(photo, annot):
    salt_intensity = np.abs(np.random.normal(0.0, 0.008))
    photo = im_utils.add_salt_pepper(photo, salt_intensity)
    return photo, annot


class UNetTransformer():
    """ Data Augmentation """
    def __init__(self):
        self.color_jit = ColorJitter(brightness=0.3, contrast=0.3,
                                     saturation=0.2, hue=0.001)

    def transform(self, photo, annot):

        transforms = random.sample([elastic_transform,
                                    guassian_noise_transform,
                                    salt_pepper_transform,
                                    self.color_jit_transform], 4)

        for transform in transforms:
            if random.random() < 0.8:
                photo, annot = transform(photo, annot)

        if random.random() < 0.5:
            photo = np.fliplr(photo)
            annot = np.fliplr(annot)

        return photo, annot

    def color_jit_transform(self, photo, annot):
        # TODO check skimage docs for something cleaner to convert
        # from float to int
        photo = rescale_intensity(photo, out_range=(0, 255))
        photo = Image.fromarray((photo).astype(np.uint8), mode='RGB')
        photo = self.color_jit(photo)  # returns PIL image
        photo = img_as_float32(np.array(photo))  # return back to numpy
        return photo, annot


class TrainDataset(Dataset):
    def __init__(self, train_annot_dir, dataset_dir, in_w, out_w,
                 min_epoch_tiles=612, foreground_tile_fraction=None):
        """
        in_w and out_w are the tile size in pixels
        min_epoch_tiles: minimum number of samples per epoch
        foreground_tile_fraction: optional fixed fraction of samples drawn
            from an annotation/crop containing explicit foreground
            supervision. None adapts the crop-routing probability to
            foreground_pool / (foreground_pool + background_pool), so every
            pool member has equal expected sampling frequency. This is not an
            estimate of dataset foreground prevalence; files with both
            correction types occur in both pools.
        """
        if (foreground_tile_fraction is not None
                and not 0 <= foreground_tile_fraction <= 1):
            raise ValueError('foreground_tile_fraction must be in [0, 1]')
        self.in_w = in_w
        self.out_w = out_w
        self.train_annot_dir = train_annot_dir
        self.dataset_dir = dataset_dir
        self.augmentor = UNetTransformer()
        self.min_epoch_tiles = min_epoch_tiles
        self.foreground_tile_fraction = foreground_tile_fraction
        self.foreground_fnames = []
        self.background_fnames = []
        self.refresh_annotation_pools()

    def __len__(self):
        return max(self.min_epoch_tiles, len(ls(self.train_annot_dir)) * 2)

    def refresh_annotation_pools(self):
        """Refresh foreground/background-bearing annotation filename pools.

        The raw image collection need not be labelled or class-balanced.
        Pools are built only from sparse corrections that already exist, and
        filenames may belong to both pools when an annotation contains both
        red foreground and green background strokes.
        """
        foreground_fnames = []
        background_fnames = []
        fnames = sorted(
            fname for fname in ls(self.train_annot_dir)
            if im_utils.is_photo(fname))
        for fname in fnames:
            annot_path = os.path.join(self.train_annot_dir, fname)
            try:
                with Image.open(annot_path) as annot_image:
                    annot = np.array(annot_image)
            except (OSError, ValueError):
                # Sync-backed files can be observed while still being written.
                # Keep all previous pools intact until the next epoch retries.
                return False
            if annot.ndim < 3 or annot.shape[2] < 2:
                continue
            if np.any(annot[:, :, 0]):
                foreground_fnames.append(fname)
            if np.any(annot[:, :, 1]):
                background_fnames.append(fname)

        changed = (
            foreground_fnames != self.foreground_fnames
            or background_fnames != self.background_fnames)
        self.foreground_fnames = foreground_fnames
        self.background_fnames = background_fnames
        return changed

    def _sampling_pool(self):
        """Return (filenames, required channel) for the next training tile."""
        has_foreground = bool(self.foreground_fnames)
        has_background = bool(self.background_fnames)
        if not has_foreground and not has_background:
            raise RuntimeError('No non-empty training annotations available')

        foreground_fraction = self.effective_foreground_tile_fraction()
        if has_foreground and has_background:
            sample_foreground = random.random() < foreground_fraction
        else:
            sample_foreground = has_foreground

        if sample_foreground:
            return self.foreground_fnames, 0
        return self.background_fnames, 1

    def effective_foreground_tile_fraction(self):
        """Return the fixed or pool-size-adaptive foreground tile target."""
        if self.foreground_tile_fraction is not None:
            return self.foreground_tile_fraction

        foreground_count = len(self.foreground_fnames)
        background_count = len(self.background_fnames)
        total_count = foreground_count + background_count
        if total_count == 0:
            return 0.0
        return foreground_count / total_count

    def __getitem__(self, _):
        fnames, required_channel = self._sampling_pool()
        image, annot, fname = load_train_image_and_annot(self.dataset_dir,
                                                         self.train_annot_dir,
                                                         fnames=fnames)
        tile_pad = (self.in_w - self.out_w) // 2

        # ensures each pixel is sampled with equal chance
        im_pad_w = self.out_w + tile_pad
        padded_w = image.shape[1] + (im_pad_w * 2)
        padded_h = image.shape[0] + (im_pad_w * 2)
        padded_im = im_utils.pad(image, im_pad_w)

        # This speeds up the padding.
        annot = annot[:, :, :2]
        padded_annot = im_utils.pad(annot, im_pad_w)
        right_lim = padded_w - self.in_w
        bottom_lim = padded_h - self.in_w

        # TODO:
        # Images with less annoations will still give the same number of
        # tiles in the training procedure as images with more annotation.
        # Further empirical investigation into effects of
        # instance selection required are required.
        while True:
            x_in = math.floor(random.random() * right_lim)
            y_in = math.floor(random.random() * bottom_lim)
            annot_tile = padded_annot[y_in:y_in+self.in_w,
                                      x_in:x_in+self.in_w]
            # U-Net predicts only the central 500 pixels of a 572-pixel
            # input. Do not accept a crop merely because its annotation lies
            # in the 36-pixel context border that is removed before loss. The
            # selected correction class must itself occur in the output area;
            # otherwise a nominal foreground sample could still train only on
            # background pixels from a mixed annotation.
            output_annot = annotation_output_region(annot_tile, tile_pad)
            if np.any(output_annot[:, :, required_channel]):
                break

        im_tile = padded_im[y_in:y_in+self.in_w,
                            x_in:x_in+self.in_w]

        assert annot_tile.shape == (self.in_w, self.in_w, 2), (
            f" shape is {annot_tile.shape} for tile from {fname}")

        assert im_tile.shape == (self.in_w, self.in_w, 3), (
            f" shape is {im_tile.shape} for tile from {fname}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        im_tile, annot_tile = self.augmentor.transform(im_tile, annot_tile)
        im_tile = im_utils.normalize_tile(im_tile)

        foreground = np.array(annot_tile)[:, :, 0]
        background = np.array(annot_tile)[:, :, 1]

        # Annotation is cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        # When tile_pad=0 (e.g. RETFound, in_w==out_w) no crop is needed —
        # avoid [0:-0] which would produce an empty slice.
        if tile_pad > 0:
            foreground = foreground[tile_pad:-tile_pad, tile_pad:-tile_pad]
            background = background[tile_pad:-tile_pad, tile_pad:-tile_pad]

        # Sparse-supervision invariant: every pixel is either foreground,
        # background, or untouched — never both. The painter enforces this
        # because each pixel of the annotation PNG holds one RGBA value, but
        # we still assert it here to catch hand-edited files or any future
        # code path that merges the model's prediction into the annotation
        # (which would silently produce mask>1 and double-weight pixels).
        fg_bool = foreground.astype(bool)
        bg_bool = background.astype(bool)
        assert not np.any(fg_bool & bg_bool), \
            f"Annotation has overlapping fg/bg pixels: {fname}"
        # 1 = supervised pixel (user marked fg or bg); 0 = untouched / ignored.
        mask = np.logical_or(fg_bool, bg_bool).astype(np.float32)
        mask = torch.from_numpy(mask)
        foreground = foreground.astype(np.int64)
        foreground = torch.from_numpy(foreground)
        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, foreground, mask
