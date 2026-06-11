"""
FunduSegmenter segmentation model adapter.

⚠️  PLACEHOLDER / STUB — THIS IS NOT THE REAL FUNDUSEGMENTER MODEL.

This adapter exists only so that ``fundusegmenter`` can be selected as a
``--model-type`` and wired end-to-end through training, validation, and
inference. It does **not** implement the FunduSegmenter architecture and
loads **no** FunduSegmenter pretrained weights. Internally it is just the
project's ``UNetGNRes`` with random initialisation, so today
``--model-type fundusegmenter`` is functionally identical to
``--model-type unet`` — do not interpret its results as a real
FunduSegmenter model.

``checkpoint_path`` is accepted for interface parity but is ignored.

When the real model is added, replace ``self.backbone`` with the actual
FunduSegmenter network and implement weight loading; the public contract
below should stay the same.

Public constructor:
    FunduSegmenter(num_classes=2, checkpoint_path=None)

Output contract:
    forward(x) -> logits with shape (B, num_classes, H, W)
"""

import warnings

import torch
import torch.nn as nn

from unet import UNetGNRes


class FunduSegmenter(nn.Module):
    """
    PLACEHOLDER wrapper around ``UNetGNRes`` — **not** the FunduSegmenter model.

    This is a plain U-Net under a different name, kept only so the
    ``fundusegmenter`` model type is wired through the app. No FunduSegmenter
    architecture or pretrained weights are used; results are identical to
    ``--model-type unet`` until the real network is implemented.
    """

    def __init__(self, num_classes: int = 2, checkpoint_path=None):
        super().__init__()
        if num_classes != 2:
            raise ValueError("FunduSegmenter currently supports num_classes=2 only.")
        warnings.warn(
            "FunduSegmenter is a PLACEHOLDER: it is currently a plain UNetGNRes "
            "with random weights (no FunduSegmenter architecture or pretrained "
            "weights). '--model-type fundusegmenter' is equivalent to "
            "'--model-type unet'. Do not treat its output as a real "
            "FunduSegmenter model.",
            stacklevel=2,
        )
        # checkpoint_path is accepted for interface parity but intentionally
        # unused: there are no FunduSegmenter weights to load yet.
        _ = checkpoint_path
        self.backbone = UNetGNRes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
