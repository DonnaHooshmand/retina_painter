"""
Phase-1 integration smoke test.

Goes beyond the unit tests by running the actual training-step path with
a real UNet and synthetic tiles, then comparing:

  * legacy approach: ``outputs[:, c] *= mask`` followed by unmasked loss
  * fixed approach: ``loss(..., mask=mask)``

at several untouched-fraction settings. We verify:

  1. Both approaches agree at untouched_fraction=0 (pure parity check).
  2. Legacy is always >= fixed for untouched_fraction > 0 (no leakage in
     the fixed path).
  3. The fixed-path absolute loss difference vs legacy roughly tracks
     ``0.7 * untouched_fraction`` nats per tile, which is the
     constant-CE-penalty term that the leakage was contributing.
  4. Training with the fixed loss actually descends across N steps —
     i.e. the masked loss still has usable gradients for the supervised
     pixels.

Run with the project venv (or any env with torch + numpy + the trainer
sources on the path):

    python smoke_phase1.py
"""

import os
import sys

import numpy as np
import torch
from torch.nn.functional import softmax

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

# pylint: disable=C0413
from loss import combined_loss, tversky_loss
from unet import UNetGNRes


def _make_synthetic_tile(in_w, out_w, untouched_fraction, seed=0):
    """Build a (image, label, mask) triple at the trainer's expected shapes.

    The image is random noise; the label is a checkerboard of fg/bg; the
    untouched region is the rightmost columns of the (out_w, out_w) tile.
    """
    rng = np.random.default_rng(seed)
    image = rng.standard_normal((1, 3, in_w, in_w)).astype(np.float32)
    label = np.zeros((1, out_w, out_w), dtype=np.int64)
    # Checkerboard foreground in the supervised region
    for i in range(out_w):
        for j in range(out_w):
            if (i // 8 + j // 8) % 2 == 0:
                label[0, i, j] = 1

    mask = np.ones((1, out_w, out_w), dtype=np.float32)
    untouched_cols = int(round(out_w * untouched_fraction))
    if untouched_cols > 0:
        # Mark the rightmost stripe as untouched.
        mask[:, :, -untouched_cols:] = 0.0
        # Convention used by the trainer: at untouched pixels, label is
        # also 0 (untouched and background both look like 0 in the label
        # tensor; only the mask distinguishes them).
        label[:, :, -untouched_cols:] = 0

    return (
        torch.from_numpy(image),
        torch.from_numpy(label),
        torch.from_numpy(mask),
    )


def _legacy_loss(outputs, label, mask, loss_fn):
    """Reproduce the pre-fix code path: zero logits at untouched pixels."""
    out = outputs.clone()
    out[:, 0] *= mask
    out[:, 1] *= mask
    return loss_fn(out, label)


def _fixed_loss(outputs, label, mask, loss_fn):
    """The new code path: pass the mask through to the loss."""
    return loss_fn(outputs, label, mask=mask)


def loss_value_comparison():
    print('=' * 72)
    print('1. Loss-value comparison: legacy `outputs *= mask` vs fixed `mask=` arg')
    print('=' * 72)

    in_w = 572  # UNet default
    out_w = 500
    model = UNetGNRes()
    model.eval()  # we don't need dropout/BN-mode weirdness for this comparison
    image, label, _ = _make_synthetic_tile(in_w, out_w, 0.0, seed=42)

    with torch.no_grad():
        outputs = model(image)
    print(f'  outputs shape: {tuple(outputs.shape)}, label shape: {tuple(label.shape)}')

    rows = []
    for uf in (0.0, 0.25, 0.50, 0.75, 0.90):
        _, lab, mask = _make_synthetic_tile(in_w, out_w, uf, seed=42)
        # Use the same outputs across all untouched-fractions so the only
        # variable is the masking strategy.
        legacy_combined = _legacy_loss(outputs, lab, mask, combined_loss).item()
        fixed_combined = _fixed_loss(outputs, lab, mask, combined_loss).item()
        legacy_tversky = _legacy_loss(outputs, lab, mask, tversky_loss).item()
        fixed_tversky = _fixed_loss(outputs, lab, mask, tversky_loss).item()
        rows.append((uf, legacy_combined, fixed_combined,
                     legacy_tversky, fixed_tversky))

    header = (
        'untouched_frac | combined: legacy   fixed     diff   |'
        ' tversky:  legacy   fixed     diff'
    )
    print()
    print(header)
    print('-' * len(header))
    for uf, lc, fc, lt, ft in rows:
        print(
            f'  {uf:>10.2f}    | combined: {lc:>7.4f} {fc:>7.4f}'
            f' {lc - fc:>+7.4f}  |'
            f' tversky:  {lt:>7.4f} {ft:>7.4f} {lt - ft:>+7.4f}'
        )

    # Assertions / sanity checks ---------------------------------------------
    print()
    issues = []

    # (a) parity at uf=0
    uf0 = rows[0]
    if abs(uf0[1] - uf0[2]) > 1e-4:
        issues.append(
            f'parity broken at uf=0 (combined): legacy={uf0[1]:.6f} fixed={uf0[2]:.6f}'
        )
    if abs(uf0[3] - uf0[4]) > 1e-4:
        issues.append(
            f'parity broken at uf=0 (tversky): legacy={uf0[3]:.6f} fixed={uf0[4]:.6f}'
        )

    # (b) legacy >= fixed for uf > 0 (legacy includes a leak penalty)
    for uf, lc, fc, lt, ft in rows[1:]:
        if lc < fc - 1e-4:
            issues.append(
                f'legacy combined < fixed at uf={uf}: {lc:.4f} < {fc:.4f}'
            )
        if lt < ft - 1e-4:
            issues.append(
                f'legacy tversky < fixed at uf={uf}: {lt:.4f} < {ft:.4f}'
            )

    # (c) The combined-loss diff at uf>0 should track ~0.3 * log(2) * uf
    #     (CE weight in combined_loss is 0.3, and the leak is log(2) per
    #     untouched pixel). Allow generous tolerance because Dice
    #     contributes too.
    expected_at_uf75 = 0.3 * np.log(2.0) * 0.75   # ~0.156
    actual_at_uf75 = rows[3][1] - rows[3][2]      # legacy - fixed (combined)
    print(
        f'  Expected combined-loss leak at uf=0.75: ~{expected_at_uf75:.3f} '
        f'(0.3 * ln(2) * 0.75)'
    )
    print(f'  Actual diff at uf=0.75:                 {actual_at_uf75:.3f}')
    if not (0.5 * expected_at_uf75 < actual_at_uf75 < 3.0 * expected_at_uf75):
        issues.append(
            f'combined-loss leak at uf=0.75 outside expected band '
            f'[{0.5*expected_at_uf75:.3f}, {3.0*expected_at_uf75:.3f}]: '
            f'{actual_at_uf75:.3f}'
        )

    if issues:
        print('\n  ISSUES:')
        for msg in issues:
            print(f'    - {msg}')
        return False
    print('\n  All comparison checks passed.')
    return True


def descent_check():
    print('\n' + '=' * 72)
    print('2. Descent check: 30 training steps with the fixed masked loss')
    print('=' * 72)

    in_w = 572
    out_w = 500
    model = UNetGNRes()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    losses_combined = []
    losses_tversky = []

    for step in range(30):
        # 50% untouched in every tile — non-trivial mask coverage.
        image, label, mask = _make_synthetic_tile(in_w, out_w, 0.50,
                                                  seed=step)
        optimizer.zero_grad()
        outputs = model(image)
        loss = combined_loss(outputs, label, mask=mask)
        loss.backward()
        optimizer.step()
        losses_combined.append(loss.item())

    # Reset model and try tversky too
    model = UNetGNRes()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(30):
        image, label, mask = _make_synthetic_tile(in_w, out_w, 0.50,
                                                  seed=step)
        optimizer.zero_grad()
        outputs = model(image)
        loss = tversky_loss(outputs, label.long(), mask=mask)
        loss.backward()
        optimizer.step()
        losses_tversky.append(loss.item())

    def summarize(name, losses):
        first5 = sum(losses[:5]) / 5
        last5 = sum(losses[-5:]) / 5
        print(
            f'  {name:>10}: first5_avg={first5:.4f}  last5_avg={last5:.4f}  '
            f'delta={first5 - last5:+.4f}'
        )
        return last5 < first5

    print()
    okc = summarize('combined', losses_combined)
    okt = summarize('tversky', losses_tversky)

    if not (okc and okt):
        print('\n  ISSUE: at least one loss did not descend over 30 steps.')
        return False
    print('\n  Both losses descended.')
    return True


def gradient_isolation_check():
    print('\n' + '=' * 72)
    print('3. End-to-end gradient isolation: untouched pixels get zero grad')
    print('=' * 72)
    in_w = 572
    out_w = 500
    model = UNetGNRes()
    model.train()

    image, label, mask = _make_synthetic_tile(in_w, out_w, 0.50, seed=7)
    image.requires_grad_(False)
    outputs = model(image)
    outputs.retain_grad()
    loss = combined_loss(outputs, label, mask=mask)
    loss.backward()

    # Untouched columns: rightmost out_w * 0.5 = 250 columns
    untouched_cols = int(round(out_w * 0.50))
    assert outputs.grad is not None, 'no grad on outputs'
    grad_untouched = outputs.grad[:, :, :, -untouched_cols:]
    grad_supervised = outputs.grad[:, :, :, :-untouched_cols]

    max_untouched = grad_untouched.abs().max().item()
    sum_supervised = grad_supervised.abs().sum().item()

    print(f'  max |grad| in untouched output region: {max_untouched:.3e}')
    print(f'  sum |grad| in supervised output region: {sum_supervised:.3e}')

    if max_untouched > 1e-7:
        print('\n  ISSUE: gradient leaked into the untouched output region.')
        return False
    if sum_supervised < 1e-6:
        print('\n  ISSUE: supervised region produced no gradient — degenerate.')
        return False
    print('\n  Gradient isolation looks correct.')
    return True


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    results = [
        loss_value_comparison(),
        descent_check(),
        gradient_isolation_check(),
    ]
    print('\n' + '=' * 72)
    if all(results):
        print('SMOKE TEST PASSED')
        return 0
    print('SMOKE TEST FAILED — see above.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
