"""
Unit tests for metrics.get_metrics.

These lock in two correctness fixes:
  - When there are no true positives, precision/recall/F1/IoU are 0.0 (not NaN),
    so model-selection / early-stopping comparisons stay well-defined and a
    "predicts nothing yet" model is distinguishable from a plateaued one.
  - When no pixels are defined at all (total == 0), get_metrics does not raise
    (accuracy/true_mean/pred_mean fall back to NaN instead of dividing by zero).
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from metrics import get_metrics


class TestGetMetrics:
    def test_normal_case(self):
        m = get_metrics(tp=5, fp=1, tn=90, fn=4, defined_sum=100, duration=1.0)
        assert math.isclose(m['precision'], 5 / 6, rel_tol=1e-6)
        assert math.isclose(m['recall'], 5 / 9, rel_tol=1e-6)
        assert math.isclose(m['f1'], 2 * (5 / 6) * (5 / 9) / ((5 / 6) + (5 / 9)),
                            rel_tol=1e-6)
        assert math.isclose(m['accuracy'], 0.95, rel_tol=1e-6)

    def test_no_true_positives_is_zero_not_nan(self):
        # Model predicted nothing useful (tp == 0) but there are real positives.
        m = get_metrics(tp=0, fp=3, tn=90, fn=7, defined_sum=100, duration=1.0)
        for k in ('precision', 'recall', 'f1', 'iou'):
            assert m[k] == 0.0, f'{k} should be 0.0, got {m[k]}'
            assert not math.isnan(m[k]), f'{k} should not be NaN'

    def test_all_zero_counts_does_not_crash(self):
        # No defined pixels anywhere -> total == 0; must not raise.
        m = get_metrics(tp=0, fp=0, tn=0, fn=0, defined_sum=0, duration=0.0)
        assert math.isnan(m['accuracy'])
        assert math.isnan(m['true_mean'])
        assert math.isnan(m['pred_mean'])
        assert m['f1'] == 0.0

    def test_loss_is_passed_through(self):
        m = get_metrics(tp=1, fp=0, tn=1, fn=0, defined_sum=2, duration=0.0,
                        loss=0.123)
        assert math.isclose(m['loss'], 0.123, rel_tol=1e-9)

    def test_loss_defaults_to_nan(self):
        m = get_metrics(tp=1, fp=0, tn=1, fn=0, defined_sum=2, duration=0.0)
        assert math.isnan(m['loss'])
