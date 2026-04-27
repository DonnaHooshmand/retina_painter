"""
Phase-1 GPU smoke test — RETFound + RETFound-RFA.

This script exercises the full training-step path (forward, masked loss,
backward) for the two models that actually matter to RetinaPainter on the
GPU:

  * RETFoundSeg          (model_type='retfound')
  * RETFoundSegRFA       (model_type='retfound_rfa')

It uses **random-initialized encoder weights** so it does not need the
3.95 GB RETFound checkpoint. The point is to verify the masked-loss path
behaves correctly with the real architectures (correct shapes, correct
device, correct loss + gradient isolation) — not to verify the encoder
is well-trained.

What this checks for each model
-------------------------------
  1. Forward + masked loss runs without error at the trainer's expected
     shapes (224x224, in_w == out_w, tile_pad=0).
  2. Loss-value parity at untouched_fraction=0 between the legacy
     `outputs *= mask` approach and the new `mask=` argument.
  3. Loss leak grows monotonically with untouched_fraction in the
     legacy path, and is zero in the fixed path.
  4. Gradients on output logits at untouched pixels are exactly 0.
  5. 10 optimizer steps with the masked loss actually descend.

Usage
-----
On the GPU machine with the trainer venv activated and trainer/src on
the path:

    cd trainer/tests
    python smoke_phase1_gpu.py

Optional: ``--use-real-weights`` to load the real RETFound checkpoint
(slow, takes 2–5 min). Default is random init.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

# pylint: disable=C0413
from loss import combined_loss, tversky_loss


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda'), 'cuda'
    if torch.backends.mps.is_available():
        return torch.device('mps'), 'mps'
    return torch.device('cpu'), 'cpu'


def make_synthetic_batch(bs, in_w, untouched_fraction, device, seed=0):
    """A batch shaped like what the RetinaPainter dataloader produces.

    Returns (image, label, mask) on the given device with shapes:
      image: (bs, 3, in_w, in_w)
      label: (bs, in_w, in_w)
      mask:  (bs, in_w, in_w)

    The untouched region is the rightmost columns. Label has a checker
    pattern in the supervised region; untouched pixels have label=0.
    """
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, (bs, 3, in_w, in_w)).astype(np.float32)
    label = np.zeros((bs, in_w, in_w), dtype=np.int64)
    for i in range(in_w):
        for j in range(in_w):
            if (i // 16 + j // 16) % 2 == 0:
                label[:, i, j] = 1

    mask = np.ones((bs, in_w, in_w), dtype=np.float32)
    untouched_cols = int(round(in_w * untouched_fraction))
    if untouched_cols > 0:
        mask[:, :, -untouched_cols:] = 0.0
        label[:, :, -untouched_cols:] = 0

    return (
        torch.from_numpy(image).to(device),
        torch.from_numpy(label).to(device),
        torch.from_numpy(mask).to(device),
    )


def build_retfound(model_type, use_real_weights, device):
    """Build either RETFoundSeg or RETFoundSegRFA on the given device."""
    if model_type == 'retfound':
        from retfound_model import RETFoundSeg, download_retfound_weights
        model = RETFoundSeg()
    elif model_type == 'retfound_rfa':
        from retfound_rfa_model import RETFoundSegRFA
        from retfound_model import download_retfound_weights
        model = RETFoundSegRFA()
        # Match trainer.py: retfound_rfa freezes the first 21 ViT blocks.
        model.freeze_encoder_blocks(21)
    else:
        raise ValueError(model_type)

    if use_real_weights:
        print(f'  Loading real RETFound weights (slow, 2-5 min) ...',
              flush=True)
        ckpt_path = download_retfound_weights()
        # Both models load the same MAE checkpoint.
        sd = torch.load(ckpt_path, map_location='cpu')
        # The MAE checkpoint format: weights live under "model" key.
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        # Encoder-only load: ignore the MAE decoder keys.
        encoder_state = {k: v for k, v in sd.items()
                         if not k.startswith('decoder')
                         and 'mask_token' not in k}
        missing, unexpected = model.encoder.load_state_dict(
            encoder_state, strict=False
        )
        print(f'  loaded encoder; missing={len(missing)} '
              f'unexpected={len(unexpected)}', flush=True)

    return model.to(device)


def legacy_loss_value(outputs, label, mask, loss_fn):
    """Pre-fix code path: zero logits at untouched pixels, unmasked loss."""
    out = outputs.clone()
    out[:, 0] *= mask
    out[:, 1] *= mask
    return loss_fn(out, label).item()


def fixed_loss_value(outputs, label, mask, loss_fn):
    """Post-fix code path: pass mask through to loss."""
    return loss_fn(outputs, label, mask=mask).item()


def smoke_one_model(model_type, model, device, loss_fn, loss_name):
    print()
    print('=' * 72)
    print(f'Smoke test: {model_type} ({loss_name})  device={device.type}')
    print('=' * 72)

    in_w = 224  # RETFound expects 224x224 (in_w == out_w, tile_pad=0)
    bs = 2

    issues = []

    # ---- (1) forward + masked loss runs ------------------------------------
    image, label, mask = make_synthetic_batch(bs, in_w, 0.50, device, seed=0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    print(f'  forward OK; outputs shape: {tuple(outputs.shape)}')
    assert outputs.shape == (bs, 2, in_w, in_w), outputs.shape

    # ---- (2,3) loss-value comparison across untouched fractions -----------
    rows = []
    for uf in (0.0, 0.25, 0.50, 0.75):
        _, lab, msk = make_synthetic_batch(bs, in_w, uf, device, seed=0)
        legacy = legacy_loss_value(outputs, lab, msk, loss_fn)
        fixed = fixed_loss_value(outputs, lab, msk, loss_fn)
        rows.append((uf, legacy, fixed))

    print()
    print(f'    untouched_frac | {loss_name}: legacy   fixed     diff')
    print('    -' + '-' * 60)
    for uf, l, f in rows:
        print(f'    {uf:>10.2f}    | {loss_name}: {l:>7.4f} {f:>7.4f} '
              f'{l - f:>+7.4f}')

    if abs(rows[0][1] - rows[0][2]) > 1e-3:
        issues.append(
            f'parity broken at uf=0: legacy={rows[0][1]:.6f} '
            f'fixed={rows[0][2]:.6f}'
        )
    for uf, l, f in rows[1:]:
        if l < f - 1e-3:
            issues.append(
                f'legacy < fixed at uf={uf}: {l:.4f} < {f:.4f}'
            )
    diffs = [l - f for _, l, f in rows]
    if not all(diffs[i] <= diffs[i + 1] + 1e-4 for i in range(len(diffs) - 1)):
        issues.append(f'leak not monotone in untouched_fraction: {diffs}')

    # ---- (4) gradient isolation -------------------------------------------
    model.train()
    image, label, mask = make_synthetic_batch(bs, in_w, 0.50, device, seed=1)
    outputs = model(image)
    outputs.retain_grad()
    loss = loss_fn(outputs, label, mask=mask)
    loss.backward()

    untouched_cols = in_w // 2
    grad_untouched = outputs.grad[:, :, :, -untouched_cols:]
    grad_supervised = outputs.grad[:, :, :, :-untouched_cols]
    max_untouched = grad_untouched.abs().max().item()
    sum_supervised = grad_supervised.abs().sum().item()
    print()
    print(f'  max |grad| untouched-output:  {max_untouched:.3e}')
    print(f'  sum |grad| supervised-output: {sum_supervised:.3e}')
    if max_untouched > 1e-6:
        issues.append(
            f'gradient leaked into untouched region: max={max_untouched:.3e}'
        )
    if sum_supervised < 1e-6:
        issues.append(
            f'supervised region produced no gradient: sum={sum_supervised:.3e}'
        )

    # ---- (5) 10 optimizer steps must descend ------------------------------
    # Match trainer.py: AdamW for retfound_rfa, SGD otherwise.
    if model_type == 'retfound_rfa':
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    losses = []
    for step in range(10):
        image, label, mask = make_synthetic_batch(bs, in_w, 0.50, device,
                                                  seed=100 + step)
        opt.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, label, mask=mask)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first3 = sum(losses[:3]) / 3
    last3 = sum(losses[-3:]) / 3
    print()
    print(f'  10-step descent: first3_avg={first3:.4f}  last3_avg={last3:.4f} '
          f' delta={first3 - last3:+.4f}')
    if not (last3 < first3):
        issues.append(
            f'loss did not descend over 10 steps: '
            f'first3={first3:.4f} last3={last3:.4f}'
        )

    if issues:
        print('\n  ISSUES:')
        for msg in issues:
            print(f'    - {msg}')
        return False
    print('\n  PASSED')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-real-weights', action='store_true',
                    help='Load real RETFound checkpoint (slow, 2-5 min).')
    ap.add_argument('--only', choices=['retfound', 'retfound_rfa'],
                    default=None,
                    help='Only run one of the two models.')
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device, dev_name = pick_device()
    print(f'device: {dev_name}')
    if dev_name == 'cuda':
        print(f'  cuda device: {torch.cuda.get_device_name(0)}')
    elif dev_name == 'cpu':
        print('  WARNING: running on CPU. RETFound is large; this is slow.')

    targets = (['retfound'] if args.only == 'retfound'
               else ['retfound_rfa'] if args.only == 'retfound_rfa'
               else ['retfound', 'retfound_rfa'])

    results = []
    for model_type in targets:
        t0 = time.time()
        print(f'\n[building {model_type}] (real_weights={args.use_real_weights})')
        model = build_retfound(model_type, args.use_real_weights, device)
        print(f'  built in {time.time() - t0:.1f}s '
              f'({sum(p.numel() for p in model.parameters()):,} params)')

        loss_fn = (tversky_loss if model_type == 'retfound_rfa'
                   else combined_loss)
        loss_name = ('tversky' if model_type == 'retfound_rfa'
                     else 'combined')
        results.append(smoke_one_model(model_type, model, device,
                                       loss_fn, loss_name))
        # Free GPU memory between models.
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print('\n' + '=' * 72)
    if all(results):
        print('GPU SMOKE TEST PASSED for all targeted models')
        return 0
    print('GPU SMOKE TEST FAILED — see above.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
