"""
Unit tests for the FunduSegmenter model adapter.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _make_batch(batch_size=1, channels=3, height=572, width=572):
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32)


class TestFunduSegmenter:
    def test_import(self):
        from fundusegmenter_model import FunduSegmenter  # noqa: F401

    def test_forward_shape(self):
        from fundusegmenter_model import FunduSegmenter

        model = FunduSegmenter(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch())
        assert out.shape == (1, 2, 500, 500), out.shape

    def test_no_nan_output(self):
        from fundusegmenter_model import FunduSegmenter

        model = FunduSegmenter(num_classes=2, checkpoint_path=None)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch())
        assert not torch.isnan(out).any(), "NaN found in FunduSegmenter output"


class TestModelUtilsIntegration:
    def test_build_model_branch(self):
        import model_utils

        model = model_utils._build_model(model_type='fundusegmenter')
        assert model.__class__.__name__ == 'FunduSegmenter'
