"""Opt-in local Qt-to-GPU-to-export smoke with synthetic images only.

Runs the real trainer CLI and painter GUI methods in separate processes.
No user settings or clinical sync directories are read by this harness.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for(predicate, process, timeout, description):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        if process.poll() is not None:
            raise RuntimeError('Trainer stopped before ' + description)
        time.sleep(0.25)
    raise TimeoutError(description)


def send(sync, name, payload):
    # Use the exact painter IPC writer, without importing its GUI modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location('painter_ipc', REPO / 'painter/src/main/python/instructions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return Path(module.send_instruction(name, payload, sync / 'instructions', sync))


def run(sync, timeout=900):
    sync = sync.resolve()
    if not sync.name.startswith('synthetic_') or sync.exists():
        raise ValueError('Use a new directory whose name begins synthetic_')
    sync.mkdir(parents=True)
    console = sync / 'trainer_console.log'
    gui_script = REPO / 'painter/tests/smoke_annotation_session.py'
    painter_python = REPO / ('painter/env/Scripts/python.exe' if os.name == 'nt' else 'painter/env/bin/python')
    command = [sys.executable, '-u', str(REPO / 'trainer/src/main.py'),
               '--syncdir', str(sync), '--model-type', 'unet', '--patchsize', '572',
               '--maxworkers', '2', '--maxbatchsize', '2']
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    started = time.monotonic()
    with console.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, cwd=REPO / 'trainer/src', stdout=output,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=flags)
        try:
            wait_for(lambda: 'Started main loop' in console.read_text(encoding='utf-8', errors='replace'),
                     process, 90, 'trainer startup')
            print('Trainer CLI ready in isolated synthetic sync directory', flush=True)
            with (sync / 'gui_prepare_console.log').open('w', encoding='utf-8') as log:
                subprocess.run([str(painter_python), str(gui_script), '--phase', 'prepare', '--syncdir', str(sync)],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=90, creationflags=flags)
            print('GUI saved ten reviews and sent automatic training instruction', flush=True)
            project = sync / 'projects/synthetic_session'
            segments = project / 'segmentations'
            artifacts = sync / 'smoke_artifacts'
            artifacts.mkdir()
            probe_names = ['synthetic_11.png', 'synthetic_12.png']
            wait_for(lambda: all((segments / name).is_file() for name in probe_names),
                     process, 120, 'initial prediction files')
            before = {}
            for name in probe_names:
                with Image.open(segments / name) as im:
                    before[name] = np.array(im)
                Image.fromarray(before[name]).save(artifacts / ('before_' + name))
            initial_models = {p.name: sha(p) for p in (project / 'models').glob('*.pkl')}

            def promoted():
                text = console.read_text(encoding='utf-8', errors='replace')
                return 'UI checkpoint promoted:' in text and 'epoch validation duration' in text

            wait_for(promoted, process, timeout, 'completed training epoch and promoted checkpoint')
            stop = send(sync, 'stop_training', {})
            wait_for(lambda: (sync / 'executed_instructions' / stop.name).is_file(), process, 90, 'stop instruction')
            print('Full training epoch and validation completed; durable checkpoint promoted', flush=True)
            # Existing predictions are intentionally immutable until the client
            # asks for a fresh one. Move only these two owned synthetic files.
            for name in probe_names:
                os.replace(segments / name, artifacts / ('initial_live_' + name))
            request = send(sync, 'segment', {
                'dataset_dir': str(sync / 'datasets/synthetic/images'),
                'seg_dir': str(segments), 'model_dir': str(project / 'models'),
                'file_names': probe_names, 'model_type': 'unet', 'model_seed': 0,
            })
            wait_for(lambda: all((segments / name).is_file() for name in probe_names),
                     process, 120, 'trained prediction files')
            changed, foreground = {}, {}
            for name in probe_names:
                with Image.open(segments / name) as im:
                    after = np.array(im)
                assert after.shape == before[name].shape == (600, 600, 4)
                changed[name] = int(np.count_nonzero((before[name][:, :, 3] > 0) != (after[:, :, 3] > 0)))
                foreground[name] = int(np.count_nonzero(after[:, :, 3]))
                assert changed[name] > 0 and foreground[name] > 0
            query = send(sync, 'trainer_status', {})
            wait_for(lambda: (sync / 'executed_instructions' / query.name).is_file(), process, 30, 'status response')
            status = json.loads((sync / 'trainer_status.json').read_text())
            assert status['training'] is False and status['model_type'] == 'unet'
            assert not list((sync / 'failed_instructions').iterdir())
            current_models = {p.name: sha(p) for p in (project / 'models').glob('*.pkl')}
            assert current_models != initial_models
            print('Fresh trained predictions differ and contain foreground; trainer stop/status verified', flush=True)
            with (sync / 'gui_verify_console.log').open('w', encoding='utf-8') as log:
                subprocess.run([str(painter_python), str(gui_script), '--phase', 'verify', '--syncdir', str(sync)],
                               cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=90, creationflags=flags)
            gui_prepare = json.loads((sync / 'gui_prepare_result.json').read_text())
            gui_verify = json.loads((sync / 'gui_verify_result.json').read_text())
            logs = list((project / 'logs').glob('*_train.csv'))
            metrics = []
            for path in logs:
                with path.open(newline='') as stream:
                    metrics.extend(csv.DictReader(stream))
            assert metrics and all(np.isfinite(float(r['loss'])) for r in metrics)
            result = {
                'synthetic_only': True, 'passed': True,
                'actual_trainer_cli': True, 'patch_size': 572, 'epoch_tile_minimum': 612,
                'batch_size_limit': 2, 'dataloader_workers': 2,
                'completed_training_epochs': len(metrics),
                'durable_checkpoint_promoted': True, 'prediction_changed_pixels': changed,
                'prediction_foreground_pixels': foreground, 'trainer_stopped': True,
                'failed_instruction_count': 0, 'gui_prepare': gui_prepare, 'gui_verify': gui_verify,
                'duration_seconds': round(time.monotonic() - started, 1),
            }
            (sync / 'smoke_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(json.dumps(result, indent=2), flush=True)
            return result
        finally:
            # This handle belongs only to the isolated trainer started above.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syncdir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    run(args.syncdir, args.timeout)
