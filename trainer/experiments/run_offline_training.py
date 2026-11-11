"""Train a pre-populated project to convergence without the painter.

This drives the production ``Trainer`` in-process: the same sampler, loss,
optimizer, validation, checkpoint promotion, automatic rollback and early
stopping the painter workflow uses. Only the transport differs. Instead of
polling the shared ``<syncdir>/instructions`` queue, the run is started
directly and ignores painter instructions, so

* an open painter session cannot switch the model or project mid-run,
* the shared instruction queue is never cleared or consumed, and
* several runs can train at once, one per GPU (``--gpu``).

Training stops when the trainer's own early stopping fires (``--max-epochs-
without-progress`` validation epochs with no improvement), when ``--max-epochs``
is reached, or when a file named ``STOP`` is created in the project directory.
Re-running the command resumes from the latest saved checkpoint.

Outputs, all inside the project:
    models/                     validation-best checkpoints (latest = best)
    logs/<date>_{train,cur_val,prev_val}.csv   the trainer's usual metric logs
    logs/offline_history.jsonl  one JSON record per metric row, with epoch
    logs/offline_server_log.txt the trainer's event log for this run
    logs/offline_run_summary.json

Run from any directory with the trainer virtual environment active:
    python run_offline_training.py --project D:\\RootPainterSync\\projects\\NAME --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', required=True, type=Path,
                        help='project directory containing NAME.seg_proj')
    parser.add_argument('--gpu', default=None,
                        help='CUDA device index(es) to use, e.g. 0 or 0,1 '
                             '(default: all visible devices)')
    parser.add_argument('--maxworkers', type=int, default=4,
                        help='DataLoader workers (default 4; keep <=4 on '
                             'Windows with RETFound)')
    parser.add_argument('--maxbatchsize', type=int, default=12)
    parser.add_argument('--patchsize', type=int, default=572,
                        help='U-Net input patch size')
    parser.add_argument('--loss-type', default='auto',
                        choices=['auto', 'combined', 'tversky'])
    parser.add_argument('--max-epochs-without-progress', type=int, default=60)
    parser.add_argument('--max-epochs', type=int, default=1000,
                        help='hard cap on epochs for this invocation')
    return parser.parse_args(argv)


def load_project(project_dir):
    """Return ``(settings, sync_dir, images_dir)`` for a project directory."""
    project_dir = Path(project_dir).resolve()
    project_files = sorted(project_dir.glob('*.seg_proj'))
    if len(project_files) != 1:
        raise ValueError(
            f'Expected exactly one .seg_proj file in {project_dir}')
    with open(project_files[0], encoding='utf-8') as stream:
        settings = json.load(stream)
    location = Path(settings['location'])
    sync_dir = project_dir
    for _ in location.parts:
        sync_dir = sync_dir.parent
    if (sync_dir / location).resolve() != project_dir:
        raise ValueError(
            'Project directory does not match the location recorded in its '
            '.seg_proj file')
    images_dir = (sync_dir / 'datasets' / settings['dataset']).resolve()
    if not images_dir.is_dir():
        raise ValueError(f'Project dataset directory not found: {images_dir}')
    return settings, sync_dir, images_dir


def git_commit():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=SRC_DIR, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:  # pylint: disable=broad-except
        return None


def build_offline_trainer(project_dir, sync_dir, args):
    """Construct a Trainer subclass bound to one project directory."""
    sys.path.insert(0, str(SRC_DIR))
    import trainer as trainer_module  # pylint: disable=import-outside-toplevel

    log_dir = Path(project_dir) / 'logs'

    class OfflineTrainer(trainer_module.Trainer):
        """Production training loop, detached from the painter transport."""

        epoch = 0
        history_path = log_dir / 'offline_history.jsonl'
        event_log_path = log_dir / 'offline_server_log.txt'
        stop_path = Path(project_dir) / 'STOP'
        stop_reason = None

        def check_for_instructions(self):
            # Offline runs never consume the shared painter instruction queue.
            if self.training and self.stop_path.exists():
                self.training = False
                self.stop_reason = 'stop_file'
                self.log('Training stopped by STOP file')

        def write_message(self, message):
            # Painter status messages would only pile up unread.
            return

        def log(self, message):
            with open(self.event_log_path, 'a+', encoding='utf-8') as stream:
                stream.write(f'{datetime.now()}|{time.time()}|{message}\n')

        def log_metrics(self, name, metrics):
            super().log_metrics(name, metrics)
            record = {'epoch': self.epoch, 'name': name, 'time': time.time()}
            for key, value in metrics.items():
                record[key] = (value.item() if hasattr(value, 'item')
                               else value)
            with open(self.history_path, 'a+', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')

    # Trainer.__init__ normally creates the sync folders and clears pending
    # painter instructions. An offline run must leave that queue untouched.
    with mock.patch.object(trainer_module, 'ensure_required_folders_exist',
                           lambda _sync_dir: None):
        return OfflineTrainer(
            sync_dir=str(sync_dir), patch_size=args.patchsize,
            max_workers=args.maxworkers, max_batch_size=args.maxbatchsize,
            model_type=args.model_type, loss_type=args.loss_type,
            max_epochs_without_progress=args.max_epochs_without_progress)


def main(argv=None):
    args = parse_args(argv)
    if args.gpu is not None:
        # Must be set before torch initialises CUDA.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:  # pylint: disable=broad-except
        pass

    project_dir = args.project.resolve()
    settings, sync_dir, images_dir = load_project(project_dir)
    args.model_type = settings.get('model_type') or 'unet'
    seed = settings.get('training_seed', settings.get('image_order_seed'))

    config = {
        'model_dir': str(project_dir / 'models'),
        'dataset_dir': str(images_dir),
        'train_annot_dir': str(project_dir / 'annotations' / 'train'),
        'val_annot_dir': str(project_dir / 'annotations' / 'val'),
        'seg_dir': str(project_dir / 'segmentations'),
        'log_dir': str(project_dir / 'logs'),
        'message_dir': str(project_dir / 'messages'),
        'model_type': args.model_type,
        'training_seed': seed,
    }
    for key in ('train_annot_dir', 'val_annot_dir'):
        if not any(Path(config[key]).glob('*.png')):
            raise SystemExit(f'No annotations found in {config[key]}')

    trainer = build_offline_trainer(project_dir, sync_dir, args)
    if trainer.stop_path.exists():
        raise SystemExit(f'Remove {trainer.stop_path} before starting')

    import torch  # pylint: disable=import-outside-toplevel
    import model_utils  # pylint: disable=import-outside-toplevel

    resumed_from = model_utils.get_latest_model_paths(config['model_dir'], 1)
    started = time.time()
    print(f'Offline training: project={project_dir.name} '
          f'model={args.model_type} seed={seed} '
          f'gpu={os.environ.get("CUDA_VISIBLE_DEVICES", "all")}', flush=True)
    trainer.start_training(config)

    epochs_run = 0
    while trainer.training and epochs_run < args.max_epochs:
        trainer.epoch += 1
        trainer.train_one_epoch()
        epochs_run += 1
        print(f'[offline] epoch {trainer.epoch} done; '
              f'best val loss {trainer.best_val_loss:.6f}; '
              f'{trainer.epochs_without_progress} epochs without progress; '
              f'elapsed {(time.time() - started) / 60:.1f} min', flush=True)

    if trainer.stop_reason is None:
        trainer.stop_reason = ('max_epochs' if trainer.training
                               else 'early_stopping')
    best_path = model_utils.get_latest_model_paths(config['model_dir'], 1)[0]
    summary = {
        'project': project_dir.name,
        'model_type': args.model_type,
        'loss_type': args.loss_type,
        'training_seed': seed,
        'stop_reason': trainer.stop_reason,
        'epochs_run': epochs_run,
        'epochs_without_progress': trainer.epochs_without_progress,
        'max_epochs_without_progress': args.max_epochs_without_progress,
        'best_val_loss': trainer.best_val_loss,
        'best_checkpoint': os.path.basename(best_path),
        'resumed_from': (os.path.basename(resumed_from[0])
                         if resumed_from else None),
        'batch_size': trainer.bs,
        'workers': trainer.num_workers,
        'duration_minutes': round((time.time() - started) / 60, 2),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'torch': torch.__version__,
        'devices': [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())],
    }
    summary_path = project_dir / 'logs' / 'offline_run_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
