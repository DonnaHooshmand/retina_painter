"""Read and write the mask formats used by RetinaPainter datasets.

Three on-disk encodings occur in curated datasets and must not be confused:

* **Annotations** (``annotations/train|val``): RGBA, red ``(255, 0, 0, 180)``
  is clinician foreground, green ``(0, 255, 0, 180)`` is clinician background
  and fully transparent pixels are untouched (unknown).
* **Reference masks saved as greyscale** (mode ``L``): 0 background, 255
  foreground.
* **Reference masks saved in the painter's segmentation format** (RGBA):
  foreground is cyan ``(0, 255, 255, 178)`` on a transparent background. The
  red channel is empty, so reading "channel 0" silently yields an all-background
  mask, and copying such a file into an annotation folder would turn every
  lesion pixel into a *background* stroke (its green channel is 255).

``read_binary_mask`` decodes both reference encodings. Always use it rather
than indexing a channel.
"""

from pathlib import Path

import numpy as np
from PIL import Image

FOREGROUND_RGBA = (255, 0, 0, 180)
BACKGROUND_RGBA = (0, 255, 0, 180)
SEGMENTATION_RGBA = (0, 255, 255, 178)


def read_binary_mask(path):
    """Return a boolean foreground mask from a reference or segmentation PNG.

    A pixel is foreground when it is visible (alpha > 0) and any colour
    channel is non-zero. This covers greyscale 0/255 masks and the painter's
    cyan-on-transparent segmentation format.
    """
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'))
    return rgba[:, :, :3].any(axis=2) & (rgba[:, :, 3] > 0)


def read_annotation(path):
    """Return ``(foreground, background)`` boolean masks of an annotation."""
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'))
    visible = rgba[:, :, 3] > 0
    foreground = (rgba[:, :, 0] > 0) & visible
    background = (rgba[:, :, 1] > 0) & visible
    if (foreground & background).any():
        raise ValueError(
            f'Annotation has overlapping foreground and background: '
            f'{Path(path).name}')
    return foreground, background


def write_annotation(path, foreground, background):
    """Write an annotation PNG in the painter's red/green RGBA format."""
    foreground = np.asarray(foreground, dtype=bool)
    background = np.asarray(background, dtype=bool)
    if foreground.shape != background.shape:
        raise ValueError('Foreground and background shapes differ')
    if (foreground & background).any():
        raise ValueError('Foreground and background overlap')
    rgba = np.zeros(foreground.shape + (4,), dtype=np.uint8)
    rgba[foreground] = FOREGROUND_RGBA
    rgba[background] = BACKGROUND_RGBA
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode='RGBA').save(path, format='PNG')


def write_segmentation(path, foreground):
    """Write a prediction PNG in the painter's segmentation overlay format."""
    foreground = np.asarray(foreground, dtype=bool)
    rgba = np.zeros(foreground.shape + (4,), dtype=np.uint8)
    rgba[foreground] = SEGMENTATION_RGBA
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode='RGBA').save(path, format='PNG')
