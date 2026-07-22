# Trainer test suite

Tests for the RetinaPainter trainer (the PyTorch server). Run from this directory.

## Quick start

```bash
# Activate the trainer venv first (one level up)
source ../env/bin/activate          # macOS / Linux
# or: ..\env\Scripts\activate       # Windows PowerShell

# Full unit suite — no downloads; ViT tests dominate CPU runtime
python -m pytest test_loss.py test_unet.py test_utils.py test_loss_masking.py \
                  test_retfound.py test_retfound_rfa.py test_instructions.py \
                  test_metrics.py test_fundusegmenter.py \
                  test_training_control.py -v
```

Expected: **83 passed**.

A single test:

```bash
python -m pytest test_loss_masking.py::test_combined_loss_zero_grad_on_untouched_pixels -v
```

## What each file tests

| File | Tests | What it covers |
|---|---:|---|
| `test_loss.py` | 2 | Dice-loss correctness on synthetic perfect-prediction tensors. Confirms the loss bottoms out at zero when predictions match labels. |
| `test_unet.py` | 3 | The original `UNetGNRes` architecture: forward pass shape, basic training step, training-with-mask path. |
| `test_loss_masking.py` | 18 | **Sparse-supervision masking + loss routing.** Locks in the contract that untouched pixels contribute **zero loss-value sensitivity** and **zero gradient** for both `combined_loss` and `tversky_loss`, and that extra untouched canvas cannot dilute the supervised objective. Includes parity tests, all-untouched-tile handling, default model-to-loss routing, and explicit loss-override checks for controlled ablations. |
| `test_retfound.py` | 14 | RETFound plain-decoder model (`--model-type retfound`): ViT token shape, `RETFoundSeg` forward-pass shape, strict checkpoint compatibility, gradient flow, 21/24-block encoder freezing, and a tiling smoke test that runs `unet_segment` on a synthetic 512×512 image. |
| `test_retfound_rfa.py` | 18 | RETFound + RFA-U-Net attention decoder (`--model-type retfound_rfa`): `forward_multi_features` shape, `RETFoundSegRFA` forward-pass shape, softmax correctness, no-NaN, gradient flow, encoder freezing (`freeze_encoder_blocks(21)`), Tversky-loss properties, and a tiling smoke test. |
| `test_instructions.py` | 11 | Painter→trainer instruction retry handling, UI model-type preservation, model switching, and optimizer routing. Confirms both RETFound decoders, including DataParallel-wrapped production models, use identical 21/24-block freezing and AdamW settings while U-Net retains its inherited SGD optimizer. |
| `test_metrics.py` | 5 | Metric edge cases, including no true positives, no defined pixels, and validation-loss passthrough. |
| `test_fundusegmenter.py` | 4 | Placeholder model construction, shape, finite output, and factory routing. |
| `test_training_control.py` | 8 | Trial seeding and RNG isolation, continuous-loss checkpoint promotion below the hard-F1 threshold, background-only validation, and U-Net context-border supervision. |
| `test_utils.py` | (helpers) | Not a test file — shared utilities (`get_acc`, etc.) imported by the others. Pytest collects no tests here. |

## End-to-end smoke scripts

These are **not collected by pytest** (no `test_` prefix). Run them by hand when you want a higher-confidence end-to-end check than unit tests provide.

| Script | What it does | When to run |
|---|---|---|
| `smoke_phase1.py` | Builds a `UNetGNRes`, runs the legacy-vs-fixed loss comparison across five untouched-fraction settings (0.0, 0.25, 0.50, 0.75, 0.90), confirms gradient isolation, and runs 30 SGD/AdamW steps to check loss descends. ~30s on CPU. | After any change to `loss.py`, `datasets.py`, or `trainer.py`'s loss-call block. |
| `smoke_phase1_gpu.py` | Same three checks (loss-value comparison, gradient isolation, descent) but for both `RETFoundSeg` and `RETFoundSegRFA` with random-init weights. Auto-detects CUDA / MPS / CPU. Use `--use-real-weights` to load the real RETFound checkpoint (slow, 2–5 min). Use `--only retfound` or `--only retfound_rfa` to scope. | On a GPU machine, before merging changes that touch the RETFound or RFA paths. |

Both scripts print a final `PASSED` / `FAILED` line so they're easy to read.

## Slow / opt-in tests

Not part of the standard suite — they download datasets and take a long time.

| File | What it does |
|---|---|
| `test_training.py` | Full corrective- and dense-annotation training benchmarks on real data (biopores, roots, nodules). Downloads ~GB of data from Zenodo on first run and trains for many epochs. Run with `python -m pytest test_training.py -v -s`. Use as a periodic regression check, not in everyday development. |
| `training_benchmarks.py` | Older benchmark module (not a pytest file — no `test_` prefix). Used internally by `test_training.py`. |
| `sim_benchmark/` | Simulated-user benchmark suite — a separate experimental harness with its own `DESIGN.md` and helper modules. Not part of the unit suite; see `sim_benchmark/README` style docs inside that folder. |
| `astar_annotator/` | Helper module used by `sim_benchmark/` for synthetic annotation generation. Not run directly. |

## Plotting helpers

| File | What it does |
|---|---|
| `plot_metrics.py` | Plots metrics from a benchmark CSV. |
| `run_latest_plot.sh` | Convenience script: finds the most recent benchmark output and runs `plot_metrics.py` on it. |

## How the test runner discovers things

- Pytest config lives in `trainer/pyproject.toml`. The default rootdir is `trainer/` and tests are picked up from this folder by the `test_*.py` naming convention.
- Test files import from the trainer source by adding `trainer/src/` to `sys.path` at the top of the module (look for the `test_dir = os.path.dirname(...)` block). This is why the trainer code uses unqualified imports like `from loss import combined_loss` — both the trainer and the tests run from `trainer/src/` on the path.
- `conftest.py` (none currently) would be the natural place for shared fixtures if the suite grows.

## What to do if a test fails

1. **`ModuleNotFoundError`** — venv isn't activated, or a dependency is missing. `pip install -r ../requirements.txt` plus `pip install pytest` should be enough for the unit suite.
2. **A `test_loss_masking.py` test fails** — somebody changed the masked-loss contract. Read the test docstring at the top of the file before relaxing any assertions; the contract is intentional.
3. **A `test_retfound*.py` tiling test fails with a device-mismatch error on Mac MPS** — should not happen post-Phase 1 (the fix is in `model_utils.unet_segment`). If it does, confirm you're on the `sparse-supervision-semantics` branch or later.
4. **`test_training.py` fails** — these benchmarks depend on real data and absolute accuracy thresholds. A small drop after a refactor is worth investigating but not necessarily a real regression; compare F1 against the previous commit's run.
