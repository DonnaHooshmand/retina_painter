"""Train a pre-populated project to convergence without the painter.

This drives the production ``Trainer`` in-process: the same sampler, loss,
optimizer, validation, checkpoint promotion and early stopping the painter
workflow uses. Only the transport differs. Instead of
polling the shared ``<syncdir>/instructions`` queue, the run is started
directly and ignores painter instructions, so

* an open painter session cannot switch the model or project mid-run,
* the shared instruction queue is never cleared or consumed, and
* several runs can train at once, one per GPU (``--gpu``).

Training stops when the trainer's own early stopping fires (``--max-epochs-
without-progress`` validation epochs with no improvement), when ``--max-epochs``
is reached, or when a file named ``STOP`` is created in the project directory.
Re-running the command resumes from the latest saved checkpoint.

Training is conventional, as in the original RootPainter and as the live
trainer now runs by default: the candidate trains continuously, the best
validation checkpoint is the one kept, and early stopping decides when to
stop. Automatic candidate rollback is off. ``--candidate-rollback-patience N``
turns it on for a controlled comparison: after N consecutive validation epochs
worse than the best checkpoint, training restarts from that checkpoint with a
fresh optimizer. On a fixed dataset that can stop a model from ever leaving
its first plateau, which is why it is no longer the default. ``--no-rollback``
is still accepted and now changes nothing.

The interactive trainer reloads the best checkpoint and re-scores it on the
validation set every epoch, because a clinician may have changed the
annotations in between. Offline the result is bit-identical each time, so it is
computed once per checkpoint and reused until the checkpoint or the validation
annotations change. ``--no-validation-cache`` restores the per-epoch re-scoring.

``--max-minutes`` gives a fixed training budget, for comparing what each
configuration delivers in the same wall-clock time. ``--trainer-commit`` runs
the trainer source exactly as it was at a git commit (for example the last
upstream RootPainter commit, as a baseline). That source is exported into
``<project>/trainer_src/`` and run unmodified: no validation cache, and no
options the older trainer does not have.

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
import inspect
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

SRC_DIR = Path(__file__).resolve().parent.parent / 'src'


class CheckpointHandle:
    """Stands in for the best checkpoint while its metrics are cached.

    The weights are only loaded when a rollback may need them.
    """

    def __init__(self, path, model_type):
        self.path = path
        self.model_type = model_type


def annotation_signature(annot_dir):
    """Cheap fingerprint of an annotation directory: names, sizes, mtimes."""
    entries = []
    for entry in sorted(os.scandir(annot_dir), key=lambda item: item.name):
        stat = entry.stat()
        entries.append((entry.name, stat.st_size, stat.st_mtime_ns))
    return tuple(entries)


def build_parser(description=__doc__):
    """Options shared by every runner that drives the production trainer."""
    parser = argparse.ArgumentParser(
        description=description,
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
                        choices=['auto', 'combined', 'combined_per_tile',
                                 'tversky'])
    parser.add_argument('--foreground-tile-fraction', type=float,
                        default=None,
                        help='fixed share of training crops drawn from '
                             'lesion-bearing annotations (default: the '
                             "sampler's adaptive share)")
    parser.add_argument('--max-epochs-without-progress', type=int, default=60)
    parser.add_argument('--max-epochs', type=int, default=1000,
                        help='hard cap on epochs for this invocation')
    parser.add_argument('--max-minutes', type=float, default=None,
                        help='stop after the first epoch that ends beyond '
                             'this many minutes of training (model loading '
                             'is not counted)')
    parser.add_argument('--candidate-rollback-patience', type=int,
                        default=None,
                        help='turn automatic rollback on: restore the '
                             'candidate from the best checkpoint after this '
                             'many consecutive worse validation epochs (off '
                             'by default, as in the live trainer)')
    parser.add_argument('--no-rollback', action='store_true',
                        help='never restore the candidate from the best '
                             'checkpoint. This is the default; the flag is '
                             'kept so older commands still run')
    parser.add_argument('--no-validation-cache', action='store_true',
                        help='re-score the best checkpoint on the validation '
                             'set every epoch, as the interactive trainer '
                             'does, instead of once per checkpoint')
    parser.add_argument('--trainer-commit', default=None,
                        help='run the trainer source as it was at this git '
                             'commit (e.g. the last upstream RootPainter '
                             'commit) instead of the working tree')
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def training_config(project_dir, images_dir, model_type, seed):
    """The ``start_training`` instruction the painter would send."""
    project_dir = Path(project_dir)
    return {
        'model_dir': str(project_dir / 'models'),
        'dataset_dir': str(images_dir),
        'train_annot_dir': str(project_dir / 'annotations' / 'train'),
        'val_annot_dir': str(project_dir / 'annotations' / 'val'),
        'seg_dir': str(project_dir / 'segmentations'),
        'log_dir': str(project_dir / 'logs'),
        'message_dir': str(project_dir / 'messages'),
        'model_type': model_type,
        'training_seed': seed,
    }


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


def export_trainer_source(commit, destination):
    """Extract ``trainer/src`` as it was at *commit*; return (src dir, sha)."""
    repo_root = SRC_DIR.parent.parent
    sha = subprocess.run(
        ['git', 'rev-parse', '--verify', f'{commit}^{{commit}}'],
        cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    target = Path(destination) / sha[:12]
    src_dir = target / 'trainer' / 'src'
    if not src_dir.is_dir():
        archive = subprocess.run(
            ['git', 'archive', '--format=tar', sha, 'trainer/src'],
            cwd=repo_root, capture_output=True, check=True).stdout
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(target, filter='data')
    return src_dir, sha


def seed_worker(_worker_id):
    """Seed NumPy and ``random`` in a DataLoader worker from its torch seed."""
    import random  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_older_trainer(seed):
    """Make a trainer without seed handling repeatable.

    Seeds the main process (model initialisation) and gives the trainer's
    DataLoader a seeded generator plus per-worker seeding. Only which random
    numbers are drawn changes; the training algorithm is untouched.
    """
    import functools  # pylint: disable=import-outside-toplevel
    import random  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    import trainer as trainer_module  # pylint: disable=import-outside-toplevel

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(seed)
    trainer_module.DataLoader = functools.partial(
        trainer_module.DataLoader, worker_init_fn=seed_worker,
        generator=generator)


def supported_kwargs(function, **kwargs):
    """Keep only the keyword arguments *function* accepts."""
    parameters = inspect.signature(function).parameters
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def rollback_patience(args):
    """Rollback patience for a run; ``None`` (the default) means no rollback."""
    if getattr(args, 'no_rollback', False):
        return None
    return getattr(args, 'candidate_rollback_patience', None)


def build_offline_trainer(project_dir, sync_dir, args, src_dir=SRC_DIR):
    """Construct a Trainer subclass bound to one project directory.

    *src_dir* is the trainer source to drive: the working tree by default, or
    an older export (see ``export_trainer_source``). Constructor options the
    chosen trainer does not have are left out.
    """
    sys.path.insert(0, str(src_dir))
    import trainer as trainer_module  # pylint: disable=import-outside-toplevel

    log_dir = Path(project_dir) / 'logs'

    class OfflineTrainer(trainer_module.Trainer):
        """Production training loop, detached from the painter transport."""

        epoch = 0
        history_path = log_dir / 'offline_history.jsonl'
        event_log_path = log_dir / 'offline_server_log.txt'
        stop_path = Path(project_dir) / 'STOP'
        stop_reason = None
        rollbacks = 0
        cache_best_validation = not getattr(args, 'no_validation_cache', False)
        best_validation_cache = None  # ((checkpoint, annotations), metrics)
        validation_cache_hits = 0

        def validation(self):
            """Validate as usual, scoring each best checkpoint only once."""
            if not self.cache_best_validation:
                return super().validation()
            utils = trainer_module.model_utils
            load_best, score = utils.get_prev_model, utils.get_val_metrics
            loaded = {}

            def get_prev_model(model_dir, model_type='unet'):
                path = utils.get_latest_model_paths(model_dir, k=1)[0]
                annotations = annotation_signature(
                    self.train_config['val_annot_dir'])
                cached = self.best_validation_cache
                if cached is not None and cached[0] == (path, annotations):
                    return CheckpointHandle(path, model_type), path
                model, path = load_best(model_dir, model_type=model_type)
                loaded['model'] = model
                loaded['key'] = (path, annotations)
                return model, path

            def get_val_metrics(cnn, *score_args, **score_kwargs):
                if isinstance(cnn, CheckpointHandle):
                    self.validation_cache_hits += 1
                    metrics = dict(self.best_validation_cache[1])
                    metrics['duration'] = 0.0
                    return metrics
                metrics = score(cnn, *score_args, **score_kwargs)
                if cnn is loaded.get('model'):
                    self.best_validation_cache = (loaded['key'], dict(metrics))
                return metrics

            with mock.patch.object(utils, 'get_prev_model', get_prev_model), \
                    mock.patch.object(utils, 'get_val_metrics',
                                      get_val_metrics):
                return super().validation()

        def _rollback_worse_candidate(self, cur_loss, ui_loss, ui_model,
                                      ui_checkpoint_path):
            patience = getattr(self, 'candidate_rollback_patience', None)
            if (patience is not None
                    and isinstance(ui_model, CheckpointHandle)
                    and self.candidate_worse_epochs + 1 >= patience):
                # A rollback can only fire on an epoch that reaches the
                # patience, so only then are the checkpoint's weights needed.
                ui_model = trainer_module.model_utils.load_model(
                    ui_model.path, model_type=ui_model.model_type)
            rolled_back = super()._rollback_worse_candidate(
                cur_loss, ui_loss, ui_model, ui_checkpoint_path)
            self.rollbacks += int(bool(rolled_back))
            return rolled_back

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

    patience = rollback_patience(args)
    if patience is None and Path(src_dir) != SRC_DIR:
        # An older trainer source always rolls back; out of reach means off.
        patience = sys.maxsize
    # Trainer.__init__ normally creates the sync folders and clears pending
    # painter instructions. An offline run must leave that queue untouched.
    with mock.patch.object(trainer_module, 'ensure_required_folders_exist',
                           lambda _sync_dir: None):
        trainer = OfflineTrainer(**supported_kwargs(
            trainer_module.Trainer.__init__,
            sync_dir=str(sync_dir), patch_size=args.patchsize,
            max_workers=args.maxworkers, max_batch_size=args.maxbatchsize,
            model_type=args.model_type, loss_type=args.loss_type,
            max_epochs_without_progress=args.max_epochs_without_progress,
            candidate_rollback_patience=patience))
    trainer.max_epochs_without_progress = args.max_epochs_without_progress
    return trainer


def main(argv=None):
    args = parse_args(argv)
    if (args.foreground_tile_fraction is not None
            and not 0 <= args.foreground_tile_fraction <= 1):
        raise SystemExit('--foreground-tile-fraction must be between 0 and 1')
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

    config = training_config(project_dir, images_dir, args.model_type, seed)
    for key in ('train_annot_dir', 'val_annot_dir'):
        if not any(Path(config[key]).glob('*.png')):
            raise SystemExit(f'No annotations found in {config[key]}')

    src_dir, trainer_source = SRC_DIR, 'working tree'
    if args.trainer_commit:
        src_dir, trainer_source = export_trainer_source(
            args.trainer_commit, project_dir / 'trainer_src')
        # An older trainer runs exactly as it was.
        args.no_validation_cache = True
    trainer = build_offline_trainer(project_dir, sync_dir, args, src_dir)
    if trainer.stop_path.exists():
        raise SystemExit(f'Remove {trainer.stop_path} before starting')
    rollback = rollback_patience(args)
    if not hasattr(trainer, 'candidate_rollback_patience'):
        rollback = None     # a trainer source from before rollback existed

    import torch  # pylint: disable=import-outside-toplevel
    import model_utils  # pylint: disable=import-outside-toplevel

    seeding = 'trainer'
    if seed is not None and not hasattr(trainer, 'configure_training_seed'):
        seed_older_trainer(int(seed))
        seeding = 'runner'

    resumed_from = model_utils.get_latest_model_paths(config['model_dir'], 1)
    started = time.time()
    print(f'Offline training: project={project_dir.name} '
          f'model={args.model_type} seed={seed} '
          f'gpu={os.environ.get("CUDA_VISIBLE_DEVICES", "all")} '
          f'trainer_source={trainer_source} '
          f'rollback={"off" if rollback is None else rollback} '
          f'validation_cache={"off" if args.no_validation_cache else "on"}',
          flush=True)
    trainer.start_training(config)
    if args.foreground_tile_fraction is not None:
        if not hasattr(trainer.train_set, 'foreground_tile_fraction'):
            raise SystemExit('This trainer source has no fixed lesion-crop '
                             'share; drop --foreground-tile-fraction')
        trainer.train_set.foreground_tile_fraction = (
            args.foreground_tile_fraction)

    train_started = time.time()

    def budget_left():
        return (args.max_minutes is None
                or time.time() - train_started < args.max_minutes * 60)

    epochs_run = 0
    while (trainer.training and epochs_run < args.max_epochs
           and budget_left()):
        trainer.epoch += 1
        trainer.train_one_epoch()
        epochs_run += 1
        print(f'[offline] epoch {trainer.epoch} done; '
              f'best val loss {getattr(trainer, "best_val_loss", float("nan")):.6f}; '
              f'{trainer.epochs_without_progress} epochs without progress; '
              f'elapsed {(time.time() - started) / 60:.1f} min', flush=True)

    if trainer.stop_reason is None:
        if not trainer.training:
            trainer.stop_reason = 'early_stopping'
        else:
            trainer.stop_reason = ('max_epochs' if budget_left()
                                   else 'max_minutes')
    best_path = model_utils.get_latest_model_paths(config['model_dir'], 1)[0]
    summary = {
        'project': project_dir.name,
        'model_type': args.model_type,
        'loss_type': args.loss_type,
        'foreground_tile_fraction': args.foreground_tile_fraction,
        'training_seed': seed,
        'stop_reason': trainer.stop_reason,
        'epochs_run': epochs_run,
        'epochs_without_progress': trainer.epochs_without_progress,
        'max_epochs_without_progress': args.max_epochs_without_progress,
        'max_minutes': args.max_minutes,
        'training_minutes': round((time.time() - train_started) / 60, 2),
        'trainer_source': trainer_source,
        'seeded_by': seeding,
        'candidate_rollback': 'off' if rollback is None else rollback,
        'rollbacks': trainer.rollbacks,
        'best_validation_cache': ('off' if args.no_validation_cache
                                  else 'on'),
        'best_validation_cache_hits': trainer.validation_cache_hits,
        'best_val_loss': getattr(trainer, 'best_val_loss', None),
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
