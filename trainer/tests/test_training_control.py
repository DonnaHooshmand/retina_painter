"""Regression tests for seeded training and checkpoint promotion."""

import math
import os
import random
import sys

import numpy as np
import pytest
import torch


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

from datasets import annotation_output_region
import model_utils
from model_utils import (combined_validation_loss, save_if_better,
                         seeded_torch_rng)
from trainer import Trainer


def _linear_state(seed):
    with seeded_torch_rng(seed):
        model = torch.nn.Linear(4, 2)
    return {name: value.detach().clone()
            for name, value in model.state_dict().items()}


def test_seeded_model_initialization_is_repeatable_and_isolated():
    first = _linear_state(42)
    second = _linear_state(42)
    different = _linear_state(43)

    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], different[name]) for name in first)

    torch.manual_seed(7)
    expected = torch.rand(2)
    torch.manual_seed(7)
    before = torch.rand(1)
    _linear_state(99)
    after = torch.rand(1)
    assert torch.equal(torch.cat((before, after)), expected)


def test_trainer_seed_repeats_python_numpy_torch_and_loader_rng(tmp_path):
    trainer = object.__new__(Trainer)
    trainer.sync_dir = str(tmp_path)

    def draw_values():
        trainer.configure_training_seed(123)
        return (random.random(), np.random.random(), torch.rand(1),
                torch.rand(1, generator=trainer.data_generator))

    first = draw_values()
    second = draw_values()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])


def test_background_only_validation_loss_remains_informative():
    sample_count = 20
    low_fp_ce = -sample_count * math.log(0.9)   # foreground p=0.1
    high_fp_ce = -sample_count * math.log(0.1)  # foreground p=0.9

    low_fp_loss = combined_validation_loss(
        0.0, 2.0, 0, low_fp_ce, sample_count)
    high_fp_loss = combined_validation_loss(
        0.0, 18.0, 0, high_fp_ce, sample_count)

    assert low_fp_loss < high_fp_loss
    assert low_fp_loss == pytest.approx(0.3 * low_fp_ce / sample_count)


def test_continuous_loss_improves_before_hard_f1_crosses_threshold():
    # A single positive at p=0.4 and p=0.1 has hard F1=0 in both cases, but
    # the continuous objective must recognize p=0.4 as the better model.
    low_probability_loss = combined_validation_loss(
        soft_inter=0.1, soft_pred_sum=0.1, foreground_defined=1,
        ce_sum=-math.log(0.1), defined_sum=1)
    improving_loss = combined_validation_loss(
        soft_inter=0.4, soft_pred_sum=0.4, foreground_defined=1,
        ce_sum=-math.log(0.4), defined_sum=1)

    assert improving_loss < low_probability_loss


def test_checkpoint_promotion_uses_lower_continuous_loss(tmp_path):
    previous_path = tmp_path / '000001_1.pkl'
    previous_path.write_bytes(b'previous')
    model = torch.nn.Linear(2, 2)

    saved_path = save_if_better(
        str(tmp_path), model, str(previous_path),
        cur_loss=0.4, prev_loss=0.5)

    assert saved_path is not None
    assert os.path.basename(saved_path).startswith('000002_')
    assert os.path.isfile(saved_path)


def test_checkpoint_is_not_promoted_for_worse_loss(tmp_path):
    previous_path = tmp_path / '000001_1.pkl'
    previous_path.write_bytes(b'previous')

    saved_path = save_if_better(
        str(tmp_path), torch.nn.Linear(2, 2), str(previous_path),
        cur_loss=0.6, prev_loss=0.5)

    assert saved_path is None
    assert len(list(tmp_path.glob('*.pkl'))) == 1


def test_unet_context_border_does_not_count_as_supervised_output():
    annot = np.zeros((572, 572, 2), dtype=np.uint8)
    annot[10, 10, 0] = 1
    assert annotation_output_region(annot, tile_pad=36).sum() == 0

    annot[100, 100, 0] = 1
    assert annotation_output_region(annot, tile_pad=36).sum() == 1


def test_retfound_has_no_discarded_annotation_border():
    annot = np.zeros((224, 224, 2), dtype=np.uint8)
    annot[0, 0, 0] = 1
    assert annotation_output_region(annot, tile_pad=0).sum() == 1


def test_same_project_segment_uses_live_training_model(tmp_path, monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.sync_dir = str(tmp_path)
    trainer.training = True
    trainer.model_type = 'unet'
    trainer.model = torch.nn.Linear(2, 2)
    model_dir = tmp_path / 'project' / 'models'
    trainer.train_config = {'model_dir': str(model_dir)}

    calls = []

    def capture_segment(in_dir, seg_dir, fname, model_paths, format_str,
                        live_model=None):
        calls.append((in_dir, seg_dir, fname, model_paths, format_str,
                      live_model))

    monkeypatch.setattr(trainer, 'segment_file', capture_segment)

    def fail_if_checkpoint_is_requested(*_args, **_kwargs):
        pytest.fail('same-project UI inference requested a saved checkpoint')

    monkeypatch.setattr(
        model_utils, 'get_latest_model_paths', fail_if_checkpoint_is_requested)

    segment_config = {
        'dataset_dir': str(tmp_path / 'dataset'),
        'seg_dir': str(tmp_path / 'segmentations'),
        'model_dir': str(model_dir),
        'model_type': 'unet',
        'file_names': ['scan.png'],
    }
    trainer.segment(segment_config)

    assert len(calls) == 1
    assert calls[0][2] == 'scan.png'
    assert calls[0][3] is None
    assert calls[0][5] is trainer.model


def test_segment_falls_back_to_saved_checkpoint_when_not_training(
        tmp_path, monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.sync_dir = str(tmp_path)
    trainer.training = False
    trainer.model_type = 'unet'
    trainer.model = None
    trainer.train_config = None

    best_path = str(tmp_path / 'models' / '000007_best.pkl')
    monkeypatch.setattr(
        model_utils, 'get_latest_model_paths',
        lambda model_dir, count: [best_path])

    calls = []

    def capture_segment(in_dir, seg_dir, fname, model_paths, format_str,
                        live_model=None):
        calls.append((model_paths, live_model))

    monkeypatch.setattr(trainer, 'segment_file', capture_segment)

    trainer.segment({
        'dataset_dir': str(tmp_path / 'dataset'),
        'seg_dir': str(tmp_path / 'segmentations'),
        'model_dir': str(tmp_path / 'models'),
        'model_type': 'unet',
        'file_names': ['scan.png'],
    })

    assert calls == [([best_path], None)]


def test_live_model_is_rejected_for_explicit_or_other_project_requests(
        tmp_path):
    trainer = object.__new__(Trainer)
    trainer.training = True
    trainer.model_type = 'unet'
    trainer.model = torch.nn.Linear(2, 2)
    trainer.train_config = {
        'model_dir': str(tmp_path / 'active' / 'models'),
    }

    assert trainer._get_live_model_for_segment({
        'model_dir': str(tmp_path / 'active' / 'models'),
        'model_paths': ['controlled.pkl'],
    }) is None
    assert trainer._get_live_model_for_segment({
        'model_dir': str(tmp_path / 'other' / 'models'),
    }) is None
    assert trainer._get_live_model_for_segment({
        'model_dir': str(tmp_path / 'active' / 'models'),
        'model_type': 'retfound',
    }) is None


@pytest.mark.parametrize('was_training', [True, False])
def test_live_inference_restores_model_mode_and_disables_grad(
        monkeypatch, was_training):
    model = torch.nn.Linear(2, 2)
    model.train(was_training)
    calls = []

    def fake_unet_segment(cnn, image, bs, in_w, out_w, threshold):
        calls.append((cnn.training, torch.is_grad_enabled()))
        return np.full(image.shape[:2], 0.75, dtype=np.float32)

    monkeypatch.setattr(model_utils, 'unet_segment', fake_unet_segment)

    result = model_utils.ensemble_segment_models(
        [model], np.zeros((4, 4, 3), dtype=np.uint8),
        bs=1, in_w=4, out_w=4)

    assert calls == [(False, False), (False, False)]
    assert model.training is was_training
    assert np.all(result == 1)
