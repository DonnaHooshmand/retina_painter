"""
FunduSegmenter segmentation model adapter.

This module provides a local, RetinaPainter-compatible model class so
FunduSegmenter can be selected as a first-class model_type in training and
inference flows. The public constructor mirrors other model adapters:

    FunduSegmenter(num_classes=2, checkpoint_path=None)

Output contract:
    forward(x) -> logits with shape (B, num_classes, H, W)
"""

import torch
import torch.nn as nn

from unet import UNetGNRes


class FunduSegmenter(nn.Module):
    """
    Thin compatibility wrapper around the project's UNet backbone.

    This keeps the model contract stable while allowing a dedicated
    `fundusegmenter` model type to be wired end-to-end across the app.
    """

    def __init__(self, num_classes: int = 2, checkpoint_path=None):
        super().__init__()
        if num_classes != 2:
            raise ValueError("FunduSegmenter currently supports num_classes=2 only.")
        # The underlying model is initialized locally; if a checkpoint path is
        # provided by future workflows, loading is handled by model_utils.
        _ = checkpoint_path
        self.backbone = UNetGNRes()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
