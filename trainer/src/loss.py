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
"""

import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torch.nn.functional import binary_cross_entropy


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


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
                (0.3 * cross_entropy(predictions, labels)))
    # When no roots use only cross entropy
    # as dice is undefined.
    return 0.3 * cross_entropy(predictions, labels)



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


def tversky_loss(predictions, labels, alpha=0.7, beta=0.3,
                 smooth=1e-6, class_weights=(1.0, 2.0)):
    """
    Tversky loss for 2-class segmentation.

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
    """
    assert torch.max(labels) <= 1
    probs = softmax(predictions, dim=1)
    # One-hot encode labels to match probs shape
    num_classes = predictions.shape[1]
    labels_long = labels.long()
    # (B, H, W) -> (B, C, H, W)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels_long.unsqueeze(1), 1.0)

    weighted_sum = 0.0
    weight_total = 0.0
    for c, w in enumerate(class_weights):
        p = probs[:, c, :, :].contiguous().view(-1)
        t = one_hot[:, c, :, :].contiguous().view(-1)
        tp = (p * t).sum()
        fn = ((1 - p) * t).sum()
        fp = (p * (1 - t)).sum()
        tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
        weighted_sum = weighted_sum + float(w) * (1 - tversky)
        weight_total += float(w)
    return weighted_sum / max(weight_total, 1e-8)
