"""
Unit tests for the RETFound backbone and RETFoundSeg model.

These tests use random weights only (no network download required) so they
run fast in any environment.

Run from the trainer/tests directory:
    cd trainer/tests
    python -m pytest test_retfound.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch
from torch.nn.functional import softmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B=1, C=3, H=224, W=224):
    """Return a random [0, 1] float32 batch tensor."""
    return torch.rand(B, C, H, W, dtype=torch.float32)


# ---------------------------------------------------------------------------
# retfound_vit tests
# ---------------------------------------------------------------------------

class TestRETFoundViT:

    def test_import(self):
        from retfound_vit import build_retfound_vit  # noqa: F401

    def test_forward_features_shape(self):
        from retfound_vit import build_retfound_vit
        model = build_retfound_vit(checkpoint_path=None)
        model.eval()
        x = _make_batch(B=1)
        with torch.no_grad():
            tokens = model.forward_features(x)
        # ViT-Large, 224/16 = 14 → 196 patch tokens, embed_dim=1024
        assert tokens.shape == (1, 196, 1024), tokens.shape

    def test_forward_features_batch(self):
        from retfound_vit import build_retfound_vit
        model = build_retfound_vit(checkpoint_path=None)
        model.eval()
        x = _make_batch(B=3)
        with torch.no_grad():
            tokens = model.forward_features(x)
        assert tokens.shape == (3, 196, 1024), tokens.shape

    def test_pos_embed_not_learned(self):
        """Positional embedding should have requires_grad=False."""
        from retfound_vit import RETFoundViT
        model = RETFoundViT()
        assert not model.pos_embed.requires_grad

    def test_incomplete_checkpoint_is_rejected(self):
        from retfound_vit import _validate_encoder_checkpoint_load

        with pytest.raises(RuntimeError, match="partially loaded"):
            _validate_encoder_checkpoint_load(
                missing=['blocks.0.norm1.weight'], unexpected=[])

        with pytest.raises(RuntimeError, match="Unexpected keys"):
            _validate_encoder_checkpoint_load(
                missing=[], unexpected=['wrong_encoder.weight'])

        _validate_encoder_checkpoint_load(missing=[], unexpected=[])


# ---------------------------------------------------------------------------
# RETFoundSeg tests
# ---------------------------------------------------------------------------

class TestRETFoundSeg:

    def test_import(self):
        from retfound_model import RETFoundSeg  # noqa: F401

    def test_output_shape(self):
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.eval()
        x = _make_batch(B=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 224, 224), out.shape

    def test_output_shape_batch(self):
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.eval()
        x = _make_batch(B=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2, 224, 224), out.shape

    def test_softmax_sums_to_one(self):
        """After softmax the class probabilities at each pixel should sum to 1."""
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.eval()
        x = _make_batch(B=1)
        with torch.no_grad():
            logits = model(x)
            probs = softmax(logits, dim=1)
        sums = probs.sum(dim=1)   # (1, 224, 224)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_imagenet_buffers_registered(self):
        """ImageNet mean/std should be registered as buffers (not parameters)."""
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        buf_names = dict(model.named_buffers())
        assert '_imagenet_mean' in buf_names
        assert '_imagenet_std'  in buf_names

    def test_gradient_flows_to_decoder(self):
        """A backward pass should produce gradients in the decoder."""
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.train()
        x = _make_batch(B=1)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least one decoder parameter has a gradient
        decoder_grads = [
            p.grad for p in model.decoder.parameters() if p.grad is not None
        ]
        assert len(decoder_grads) > 0, "No gradients in decoder"

    def test_freeze_encoder_blocks(self):
        """Plain RETFound uses the same 21/24 block policy as RFA."""
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.freeze_encoder_blocks(21)

        assert all(
            not parameter.requires_grad
            for block in model.encoder.blocks[:21]
            for parameter in block.parameters()
        )
        assert all(
            parameter.requires_grad
            for block in model.encoder.blocks[21:]
            for parameter in block.parameters()
        )
        assert all(parameter.requires_grad
                   for parameter in model.decoder.parameters())

    def test_no_nan_in_output(self):
        from retfound_model import RETFoundSeg
        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model.eval()
        x = _make_batch(B=1)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "NaN found in output"


# ---------------------------------------------------------------------------
# Tiling smoke test (uses model_utils.unet_segment with RETFoundSeg)
# ---------------------------------------------------------------------------

class TestRETFoundTiling:

    def test_segment_larger_image(self):
        """unet_segment with in_w=out_w=224 should handle an image larger than 224."""
        pytest.importorskip("skimage", reason="scikit-image not installed")

        # Insert trainer/src onto path so model_utils can be imported
        trainer_src = os.path.join(os.path.dirname(__file__), '..', 'src')
        if trainer_src not in sys.path:
            sys.path.insert(0, trainer_src)

        from retfound_model import RETFoundSeg
        import model_utils

        model = RETFoundSeg(num_classes=2, checkpoint_path=None)
        model = torch.nn.DataParallel(model)
        model.eval()

        # Synthetic 512×512 RGB image (values in [0, 1])
        image = np.random.rand(512, 512, 3).astype(np.float32)

        predicted = model_utils.unet_segment(
            model, image, bs=1, in_w=224, out_w=224, threshold=0.5
        )
        assert predicted.shape == (512, 512), predicted.shape
