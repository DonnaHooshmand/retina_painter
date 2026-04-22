"""
Unit tests for the RFA-U-Net backbone (RETFoundSegRFA) and Tversky loss.

All tests use random weights — no checkpoint download required.

Run from the trainer/tests directory:
    cd trainer/tests
    python -m pytest test_retfound_rfa.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch.nn.functional import softmax
import pytest


def _batch(B=1, C=3, H=224, W=224):
    return torch.rand(B, C, H, W, dtype=torch.float32)


# ---------------------------------------------------------------------------
# forward_multi_features
# ---------------------------------------------------------------------------

class TestMultiFeatures:

    def test_returns_four_tensors(self):
        from retfound_vit import build_retfound_vit
        model = build_retfound_vit(checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model.forward_multi_features(_batch())
        assert len(out) == 4

    def test_each_skip_shape(self):
        from retfound_vit import build_retfound_vit
        model = build_retfound_vit(checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            skips = model.forward_multi_features(_batch(B=2))
        for s in skips:
            assert s.shape == (2, 196, 1024), s.shape

    def test_custom_indices(self):
        from retfound_vit import build_retfound_vit
        model = build_retfound_vit(checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model.forward_multi_features(_batch(), indices=(0, 6, 12, 23))
        assert len(out) == 4


# ---------------------------------------------------------------------------
# RETFoundSegRFA
# ---------------------------------------------------------------------------

class TestRETFoundSegRFA:

    def test_import(self):
        from retfound_rfa_model import RETFoundSegRFA  # noqa: F401

    def test_output_shape(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert out.shape == (1, 2, 224, 224), out.shape

    def test_output_shape_batch(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model(_batch(B=2))
        assert out.shape == (2, 2, 224, 224), out.shape

    def test_softmax_sums_to_one(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            probs = softmax(model(_batch()), dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, 224, 224), atol=1e-5)

    def test_no_nan_in_output(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model(_batch())
        assert not torch.isnan(out).any()

    def test_gradient_flows_to_decoder(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.train()
        out = model(_batch())
        out.sum().backward()
        decoder_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
        assert len(decoder_grads) > 0

    def test_imagenet_buffers_registered(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        bufs = dict(model.named_buffers())
        assert '_imagenet_mean' in bufs
        assert '_imagenet_std' in bufs

    def test_freeze_encoder_blocks(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.freeze_encoder_blocks(21)
        frozen = sum(1 for i, blk in enumerate(model.encoder.blocks)
                     if i < 21
                     for p in blk.parameters() if not p.requires_grad)
        trainable = sum(1 for i, blk in enumerate(model.encoder.blocks)
                        if i >= 21
                        for p in blk.parameters() if p.requires_grad)
        assert frozen > 0
        assert trainable > 0

    def test_freeze_does_not_affect_decoder(self):
        from retfound_rfa_model import RETFoundSegRFA
        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model.freeze_encoder_blocks(24)  # freeze all encoder blocks
        decoder_trainable = [p for p in model.decoder.parameters() if p.requires_grad]
        assert len(decoder_trainable) > 0


# ---------------------------------------------------------------------------
# Tversky loss
# ---------------------------------------------------------------------------

class TestTverskyLoss:

    def test_import(self):
        from loss import tversky_loss  # noqa: F401

    def test_scalar_output(self):
        from loss import tversky_loss
        preds = torch.randn(2, 2, 224, 224)
        labels = torch.randint(0, 2, (2, 224, 224))
        loss = tversky_loss(preds, labels)
        assert loss.ndim == 0

    def test_non_negative(self):
        from loss import tversky_loss
        preds = torch.randn(2, 2, 224, 224)
        labels = torch.randint(0, 2, (2, 224, 224))
        assert tversky_loss(preds, labels).item() >= 0

    def test_perfect_prediction_near_zero(self):
        from loss import tversky_loss
        labels = torch.zeros(1, 224, 224, dtype=torch.long)
        labels[:, 50:100, 50:100] = 1
        # Create logits strongly predicting the correct class
        preds = torch.full((1, 2, 224, 224), -10.0)
        preds[0, 0, :, :] = 10.0           # background everywhere
        preds[0, 1, 50:100, 50:100] = 10.0  # foreground in correct region
        preds[0, 0, 50:100, 50:100] = -10.0
        loss = tversky_loss(preds, labels)
        assert loss.item() < 0.05

    def test_gradients_flow(self):
        from loss import tversky_loss
        preds = torch.randn(1, 2, 224, 224, requires_grad=True)
        labels = torch.randint(0, 2, (1, 224, 224))
        tversky_loss(preds, labels).backward()
        assert preds.grad is not None


# ---------------------------------------------------------------------------
# Tiling smoke test
# ---------------------------------------------------------------------------

class TestRFATiling:

    def test_segment_larger_image(self):
        pytest.importorskip("skimage", reason="scikit-image not installed")
        import numpy as np
        from retfound_rfa_model import RETFoundSegRFA
        import model_utils

        model = RETFoundSegRFA(num_classes=2, checkpoint_path=None)
        model = torch.nn.DataParallel(model)
        model.eval()

        image = np.random.rand(512, 512, 3).astype(np.float32)
        predicted = model_utils.unet_segment(model, image, bs=1, in_w=224, out_w=224, threshold=0.5)
        assert predicted.shape == (512, 512), predicted.shape
