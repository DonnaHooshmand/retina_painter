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
The dataset returns four tensors per tile:

  * ``image``     — the photo
  * ``foreground``— the label, 1 where the clinician painted lesion,
                    0 elsewhere
  * ``mask``      — 1 where the pixel is supervised (clinician marked
                    foreground OR background), 0 where it must be ignored
                    by the loss (untouched OR unsure)
  * ``unsure``    — 1 where the clinician explicitly marked the pixel
                    as ambiguous; 0 otherwise. Already excluded from
                    ``mask`` — returned separately so the trainer can
                    log per-image difficulty / "unsure_frac" metrics.

On-disk encoding
----------------
Annotations are RGBA PNGs written by the painter. By convention:

  * channel 0 (R) = foreground brush
  * channel 1 (G) = background brush
  * channel 2 (B) = unsure brush  (Phase 2; legacy projects have this 0)
  * channel 3 (A) = pixel-painted alpha (unused at the trainer)

Backward compatibility: legacy 2-channel projects (no Unsure brush)
read with channel 2 = 0 everywhere, so ``mask`` reduces to
``foreground | background`` and behaviour is unchanged.

See ``loss.py`` and ``docs/supervision_plan.md``.
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

def elastic_transform(photo, annot):
    def_map = elastic.get_elastic_map(photo.shape,
                                      scale=random.random(),
                                      intensity=0.4 + (0.6 * random.random()))
    photo = elastic.transform_image(photo, def_map)
    # Channel 0=fg, 1=bg, 2=unsure — all three need the same elastic warp
    # so that the binary masks stay aligned with the warped image.
    annot = elastic.transform_image(annot, def_map, channels=3)
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
                 min_epoch_tiles=612):
        """
        in_w and out_w are the tile size in pixels
        min_epoch_tiles: minimum number of samples per epoch
        """
        self.in_w = in_w
        self.out_w = out_w
        self.train_annot_dir = train_annot_dir
        self.dataset_dir = dataset_dir
        self.augmentor = UNetTransformer()
        self.min_epoch_tiles = min_epoch_tiles

    def __len__(self):
        return max(self.min_epoch_tiles, len(ls(self.train_annot_dir)) * 2)

    def __getitem__(self, _):
        image, annot, fname = load_train_image_and_annot(self.dataset_dir,
                                                         self.train_annot_dir)
        tile_pad = (self.in_w - self.out_w) // 2

        # ensures each pixel is sampled with equal chance
        im_pad_w = self.out_w + tile_pad
        padded_w = image.shape[1] + (im_pad_w * 2)
        padded_h = image.shape[0] + (im_pad_w * 2)
        padded_im = im_utils.pad(image, im_pad_w)

        # Keep the first three channels: fg, bg, unsure. Drop the alpha
        # channel (channel 3) — it carries no supervision.
        annot = annot[:, :, :3]
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
            # Only count fg+bg towards "is this tile worth training on" —
            # an unsure-only tile produces a fully-masked-out loss and
            # would just waste a forward+backward pass.
            if np.sum(annot_tile[:, :, :2]) > 0:
                break

        im_tile = padded_im[y_in:y_in+self.in_w,
                            x_in:x_in+self.in_w]

        assert annot_tile.shape == (self.in_w, self.in_w, 3), (
            f" shape is {annot_tile.shape} for tile from {fname}")

        assert im_tile.shape == (self.in_w, self.in_w, 3), (
            f" shape is {im_tile.shape} for tile from {fname}")

        im_tile = img_as_float32(im_tile)
        im_tile = im_utils.normalize_tile(im_tile)
        im_tile, annot_tile = self.augmentor.transform(im_tile, annot_tile)
        im_tile = im_utils.normalize_tile(im_tile)

        foreground = np.array(annot_tile)[:, :, 0]
        background = np.array(annot_tile)[:, :, 1]
        unsure     = np.array(annot_tile)[:, :, 2]

        # Annotation is cropped post augmentation to ensure
        # elastic grid doesn't remove the edges.
        # When tile_pad=0 (e.g. RETFound, in_w==out_w) no crop is needed —
        # avoid [0:-0] which would produce an empty slice.
        if tile_pad > 0:
            foreground = foreground[tile_pad:-tile_pad, tile_pad:-tile_pad]
            background = background[tile_pad:-tile_pad, tile_pad:-tile_pad]
            unsure     = unsure[tile_pad:-tile_pad, tile_pad:-tile_pad]

        # Sparse-supervision invariant: each pixel belongs to exactly one
        # of {foreground, background, unsure, untouched}. The painter
        # enforces this because each pixel of the annotation PNG holds
        # one RGBA value, but we still assert it here to catch hand-edited
        # files or any future code path that merges layers incorrectly.
        fg_bool = foreground.astype(bool)
        bg_bool = background.astype(bool)
        un_bool = unsure.astype(bool)
        assert not np.any(fg_bool & bg_bool), \
            f"Annotation has overlapping fg/bg pixels: {fname}"
        assert not np.any(fg_bool & un_bool), \
            f"Annotation has overlapping fg/unsure pixels: {fname}"
        assert not np.any(bg_bool & un_bool), \
            f"Annotation has overlapping bg/unsure pixels: {fname}"

        # mask = 1 where the pixel is supervised (user marked fg or bg
        # AND did not mark it unsure); 0 elsewhere (untouched or unsure).
        # Untouched and unsure both contribute zero loss / zero gradient;
        # the distinction is metadata for curriculum / uncertainty work,
        # not a different loss term.
        defined = np.logical_or(fg_bool, bg_bool)
        mask = (defined & ~un_bool).astype(np.float32)
        mask = torch.from_numpy(mask)

        foreground = foreground.astype(np.int64)
        foreground = torch.from_numpy(foreground)
        unsure_t = torch.from_numpy(un_bool.astype(np.float32))

        im_tile = im_tile.astype(np.float32)
        im_tile = np.moveaxis(im_tile, -1, 0)
        im_tile = torch.from_numpy(im_tile)
        return im_tile, foreground, mask, unsure_t
