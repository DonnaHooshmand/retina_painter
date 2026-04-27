"""
Regression tests for sparse-supervision masking in the loss functions.

The contract these tests defend:

  * If ``mask[pixel] == 0`` (the clinician left the pixel untouched), then
    the logits and label at that pixel must not affect the loss value or
    its gradient. Untouched pixels are unsupervised, full stop.
  * If ``mask[pixel] == 1``, the loss should match the legacy behavior on
    fully-supervised pixels, so we don't silently regress on already
    well-trained projects.

See ``docs/supervision_plan.md`` (Phase 1).
"""

import os
import sys

import numpy as np
import pytest
import torch

# Add trainer/src to sys.path so we can import the project modules
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

# pylint: disable=C0413
from loss import combined_loss, tversky_loss


def _make_three_region_tile(h=8, w=12):
    """Build a (logits, labels, mask) triple with three explicit regions:

    - foreground region (label=1, mask=1)
    - background region (label=0, mask=1)
    - untouched region (label=0, mask=0)

    Logits are filled with deliberate values so the loss is non-trivial
    at the supervised regions.
    """
    # Tile divided into three vertical stripes of equal width.
    third = w // 3
    labels = torch.zeros(1, h, w, dtype=torch.long)
    mask = torch.zeros(1, h, w, dtype=torch.float32)

    # foreground stripe
    labels[:, :, :third] = 1
    mask[:, :, :third] = 1.0
    # background stripe
    mask[:, :, third:2 * third] = 1.0
    # untouched stripe stays mask=0, label=0

    # Logits: prefer foreground in the first stripe, prefer background
    # in the second; deliberately wrong (predicts foreground) in the
    # untouched stripe so we can detect any loss leakage.
    logits = torch.zeros(1, 2, h, w)
    logits[:, 1, :, :third] = 4.0   # confidently foreground (correct)
    logits[:, 0, :, third:2 * third] = 4.0  # confidently background (correct)
    logits[:, 1, :, 2 * third:] = 4.0  # confidently foreground (untouched)
    return logits, labels, mask


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------

def test_combined_loss_invariant_to_untouched_logits():
    """Changing logits inside the untouched region must not change loss."""
    logits, labels, mask = _make_three_region_tile()
    base = combined_loss(logits, labels, mask=mask).item()

    # Replace untouched-region logits with arbitrary garbage.
    perturbed = logits.clone()
    perturbed[:, 0, :, 8:] = -10.0
    perturbed[:, 1, :, 8:] = 7.5
    new = combined_loss(perturbed, labels, mask=mask).item()

    assert abs(new - base) < 1e-6, (
        f"untouched-pixel logits leaked into combined_loss: "
        f"{base:.6f} -> {new:.6f}"
    )


def test_combined_loss_zero_grad_on_untouched_pixels():
    """Gradient w.r.t. logits in the untouched region must be exactly zero."""
    logits, labels, mask = _make_three_region_tile()
    logits = logits.clone().requires_grad_(True)
    loss = combined_loss(logits, labels, mask=mask)
    loss.backward()

    # Untouched stripe is columns 8..end.
    untouched_grad = logits.grad[:, :, :, 8:]
    assert torch.all(untouched_grad == 0), (
        f"non-zero gradient in untouched region; max |grad| = "
        f"{untouched_grad.abs().max().item():.6e}"
    )

    # Sanity: supervised regions DO receive gradient — otherwise the
    # test passes trivially even if the loss is broken.
    supervised_grad = logits.grad[:, :, :, :8]
    assert supervised_grad.abs().sum().item() > 0, \
        "supervised pixels should produce non-zero gradient"


def test_combined_loss_matches_unmasked_when_fully_supervised():
    """Mask of all 1s should match the legacy unmasked combined_loss."""
    logits, labels, _ = _make_three_region_tile()
    full_mask = torch.ones_like(labels, dtype=torch.float32)

    masked = combined_loss(logits, labels, mask=full_mask).item()
    legacy = combined_loss(logits, labels, mask=None).item()

    assert abs(masked - legacy) < 1e-4, (
        f"masked loss with mask=ones diverges from legacy unmasked: "
        f"masked={masked:.6f} legacy={legacy:.6f}"
    )


def test_combined_loss_handles_all_untouched_tile():
    """A tile with mask=0 everywhere returns 0 with intact grad graph."""
    logits, labels, _ = _make_three_region_tile()
    logits = logits.clone().requires_grad_(True)
    mask = torch.zeros_like(labels, dtype=torch.float32)
    loss = combined_loss(logits, labels, mask=mask)
    assert loss.item() == 0.0
    # Must still be a tensor with a grad_fn so .backward() doesn't error.
    loss.backward()
    assert torch.all(logits.grad == 0), \
        "all-untouched tile should produce zero gradient everywhere"


# ---------------------------------------------------------------------------
# tversky_loss (used by retfound_rfa)
# ---------------------------------------------------------------------------

def test_tversky_loss_invariant_to_untouched_logits():
    logits, labels, mask = _make_three_region_tile()
    base = tversky_loss(logits, labels, mask=mask).item()

    perturbed = logits.clone()
    perturbed[:, 0, :, 8:] = -10.0
    perturbed[:, 1, :, 8:] = 7.5
    new = tversky_loss(perturbed, labels, mask=mask).item()

    assert abs(new - base) < 1e-6, (
        f"untouched-pixel logits leaked into tversky_loss: "
        f"{base:.6f} -> {new:.6f}"
    )


def test_tversky_loss_zero_grad_on_untouched_pixels():
    logits, labels, mask = _make_three_region_tile()
    logits = logits.clone().requires_grad_(True)
    loss = tversky_loss(logits, labels, mask=mask)
    loss.backward()

    untouched_grad = logits.grad[:, :, :, 8:]
    assert torch.all(untouched_grad == 0), (
        f"non-zero gradient in untouched region for tversky_loss; "
        f"max |grad| = {untouched_grad.abs().max().item():.6e}"
    )

    supervised_grad = logits.grad[:, :, :, :8]
    assert supervised_grad.abs().sum().item() > 0


def test_tversky_loss_matches_unmasked_when_fully_supervised():
    logits, labels, _ = _make_three_region_tile()
    full_mask = torch.ones_like(labels, dtype=torch.float32)

    masked = tversky_loss(logits, labels, mask=full_mask).item()
    legacy = tversky_loss(logits, labels, mask=None).item()

    assert abs(masked - legacy) < 1e-4, (
        f"masked tversky loss with mask=ones diverges from legacy: "
        f"masked={masked:.6f} legacy={legacy:.6f}"
    )


def test_tversky_loss_handles_all_untouched_tile():
    logits, labels, _ = _make_three_region_tile()
    logits = logits.clone().requires_grad_(True)
    mask = torch.zeros_like(labels, dtype=torch.float32)
    loss = tversky_loss(logits, labels, mask=mask)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(logits.grad == 0)


# ---------------------------------------------------------------------------
# Demonstration: legacy "outputs *= mask" trick was leaky
# ---------------------------------------------------------------------------

def test_legacy_zeroing_logits_was_leaky():
    """Documents the pre-fix bug.

    Multiplying the logits by the mask (the old approach) is *not* the
    same as masking the loss: softmax(0, 0) = (0.5, 0.5), so each
    untouched pixel still incurs a constant CE penalty. This test
    exists to make sure nobody re-introduces the old behavior thinking
    it's equivalent.
    """
    logits, labels, mask = _make_three_region_tile()

    # Legacy behavior: zero the logits at untouched pixels, then call
    # the unmasked loss.
    legacy_logits = logits.clone()
    legacy_logits[:, 0] *= mask
    legacy_logits[:, 1] *= mask
    legacy = combined_loss(legacy_logits, labels, mask=None).item()

    # Correct behavior: mask inside the loss.
    correct = combined_loss(logits, labels, mask=mask).item()

    # The two must NOT be equal — that's the whole point of the fix.
    # Legacy is larger because of the constant ~log(2) CE penalty per
    # untouched pixel and because Dice sees 0.5-prob predictions there.
    assert legacy > correct + 0.05, (
        f"legacy loss ({legacy:.4f}) should be visibly larger than "
        f"masked loss ({correct:.4f}); if they're equal, the masked "
        f"loss has regressed to legacy behavior."
    )
