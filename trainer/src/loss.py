"""
Copyright (C) 2019 Abraham George Smith

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

Sparse-supervision policy
-------------------------
RetinaPainter uses sparse corrective annotation: only pixels the clinician
has explicitly marked as foreground or background are supervised. Untouched
pixels (and, in a future revision, pixels marked as ``unsure``) must
contribute zero loss and zero gradient.

The masked loss variants in this module enforce that. Callers pass a
binary ``mask`` tensor (1 = supervised, 0 = ignored) and the loss is
computed over supervised pixels only. See ``docs/supervision_plan.md``.
"""

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy


def _safe_div(num, den):
    return num / den.clamp(min=1.0)


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def masked_dice_loss(predictions, labels, mask):
    """Dice loss restricted to supervised pixels (``mask == 1``).

    Untouched pixels contribute nothing to the intersection or union, so
    they cannot affect either the loss value or its gradient.
    """
    softmaxed = softmax(predictions, 1)
    fg_probs = softmaxed[:, 1, :, :]
    labels_f = labels.float()
    mask_f = mask.float()

    masked_preds = fg_probs * mask_f
    masked_labels = labels_f * mask_f

    intersection = torch.sum(masked_preds * masked_labels)
    union = torch.sum(masked_preds) + torch.sum(masked_labels)
    # If no supervised foreground at all, dice is undefined; return 0.
    return 1 - _safe_div(2 * intersection, union)


def combined_loss(predictions, labels, mask=None):
    """Combined Dice + 0.3 cross-entropy loss.

    If ``mask`` is provided, it is used as an ignore mask: pixels with
    ``mask == 0`` contribute zero loss and zero gradient (sparse
    supervision). If ``mask`` is ``None``, behaves like the original
    unmasked loss for backwards compatibility with old callers.
    """
    assert torch.max(labels) <= 1
    if mask is None:
        if torch.sum(labels) > 0:
            return (dice_loss(predictions, labels) +
                    (0.3 * cross_entropy(predictions, labels)))
        # When no roots use only cross entropy as dice is undefined.
        return 0.3 * cross_entropy(predictions, labels)

    mask_f = mask.float()
    mask_sum = mask_f.sum()
    # If nothing in this tile is supervised, return a zero with grad.
    if mask_sum.item() == 0:
        return (predictions.sum() * 0.0)

    # Masked CE: per-pixel CE, weighted by mask, mean over supervised pixels.
    ce_per_pixel = cross_entropy(predictions, labels, reduction='none')
    ce = _safe_div((ce_per_pixel * mask_f).sum(), mask_sum)

    # Dice is only defined when there is supervised foreground.
    if torch.sum(labels.float() * mask_f) > 0:
        return masked_dice_loss(predictions, labels, mask) + 0.3 * ce
    return 0.3 * ce



def dice_loss2(preds, labels):
    """ based on loss function from V-Net paper """
    assert torch.max(labels) <= 1
    assert torch.min(labels) >= 0
    assert torch.max(preds) <= 1
    assert torch.min(preds) >= 0

    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss2(preds, labels, mask=None):
    """ mix of dice and BCE for single-channel sigmoid output.
        Not yet used - needs benchmarking against combined_loss before switching. """
    if mask is not None:
        preds = torch.mul(preds, mask)
    assert torch.max(labels) <= 1
    cx = 0.3 * binary_cross_entropy(preds, labels)
    if torch.sum(labels) > 0:
        return dice_loss2(preds, labels) + cx
    return cx


def tversky_loss(predictions, labels, mask=None, alpha=0.7, beta=0.3,
                 smooth=1e-6, class_weights=(1.0, 2.0)):
    """
    Tversky loss for 2-class segmentation, with optional supervision mask.

    The Tversky index generalises Dice: with ``alpha=beta=0.5`` it equals
    Dice; setting ``alpha > beta`` weights false negatives more heavily,
    which is useful when the foreground class is small (as in subtle
    retinal biomarkers).

    Defaults (``alpha=0.7, beta=0.3, class_weights=(1.0, 2.0)``) match the
    RFA-U-Net reference implementation.

    Parameters
    ----------
    predictions : (B, C, H, W) float tensor
        Raw logits — softmax is applied internally.
    labels : (B, H, W) integer tensor
        Ground-truth class indices (0 = background, 1 = foreground).
    mask : (B, H, W) float tensor, optional
        Supervision mask. ``mask == 1`` means the pixel is supervised
        and contributes to the loss; ``mask == 0`` means the pixel is
        untouched / unsure and contributes zero loss and zero gradient.
        If ``None``, every pixel is treated as supervised (legacy
        behaviour).
    """
    assert torch.max(labels) <= 1
    probs = softmax(predictions, dim=1)
    # One-hot encode labels to match probs shape
    num_classes = predictions.shape[1]
    labels_long = labels.long()
    # (B, H, W) -> (B, C, H, W)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels_long.unsqueeze(1), 1.0)

    if mask is None:
        mask_f = torch.ones_like(labels, dtype=probs.dtype)
    else:
        mask_f = mask.to(probs.dtype)
        if mask_f.sum().item() == 0:
            return predictions.sum() * 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for c, w in enumerate(class_weights):
        p = probs[:, c, :, :]
        t = one_hot[:, c, :, :]
        # Restrict tp/fn/fp to supervised pixels only.
        p_m = (p * mask_f).contiguous().view(-1)
        t_m = (t * mask_f).contiguous().view(-1)
        # (1 - p) and (1 - t) should also be masked so untouched pixels
        # don't sneak into the false-negative / false-positive sums.
        one_minus_p_m = ((1 - p) * mask_f).contiguous().view(-1)
        one_minus_t_m = ((1 - t) * mask_f).contiguous().view(-1)
        tp = (p_m * t_m).sum()
        fn = (one_minus_p_m * t_m).sum()
        fp = (p_m * one_minus_t_m).sum()
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        weighted_sum = weighted_sum + float(w) * (1 - tversky)
        weight_total += float(w)
    return weighted_sum / max(weight_total, 1e-8)


def resolve_training_loss_type(model_type, loss_type='auto'):
    """Resolve an explicit or model-family-default loss selection."""
    valid_loss_types = ('auto', 'combined', 'tversky')
    if loss_type not in valid_loss_types:
        raise ValueError(
            f"Unknown loss type {loss_type!r}; expected one of "
            f"{valid_loss_types}"
        )
    if loss_type == 'auto':
        # Keep the training objective constant when the user selects a model
        # in the painter. This makes the U-Net -> RETFound -> RFA comparison
        # an architecture comparison instead of silently changing the loss.
        return 'combined'
    return loss_type


def training_loss(predictions, labels, model_type='unet', mask=None,
                  loss_type='auto'):
    """Return the intended training loss for a model family.

    Every model defaults to RootPainter's combined Dice/CE objective so
    selecting an architecture in the painter does not also change the
    optimization target. ``loss_type='tversky'`` remains available as an
    explicit controlled ablation without changing checkpoint structure.
    """
    resolved_loss_type = resolve_training_loss_type(model_type, loss_type)
    if resolved_loss_type == 'tversky':
        return tversky_loss(predictions, labels, mask=mask)
    return combined_loss(predictions, labels, mask=mask)
