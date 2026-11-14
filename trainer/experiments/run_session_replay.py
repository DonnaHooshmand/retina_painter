"""Replay an annotation session: scans arrive one at a time while training runs.

``run_offline_training.py`` trains on every annotation at once, which is the
end of a session. RetinaPainter's task is the session itself: the trainer
learns while the user works through scans it has not seen. This runner
reproduces that with annotations that already exist, so no one has to sit at
the painter:

* The scans of a cross-validation fold project that are not held out are
  taken in the project's navigation order. Scan *i* is finished
  ``i * --pace-seconds`` into the session; at that moment its annotation file,
  if the clinician painted on it, appears in ``annotations/train`` or
  ``annotations/val`` (the side the project assigned it). Scans the clinician
  left unpainted take the same time and add nothing, as in the painter.
* Training starts after ``--warmup-scans`` scans (default: the project's
  ``auto_start_training_after_reviews``, 10), as the painter starts it. From
  then on the production trainer runs exactly as it does live: new
  annotations are picked up at the start of each epoch, validation,
  checkpoint promotion, rollback and early stopping are unchanged, and the
  best checkpoint is re-scored every epoch (``--validation-cache`` turns the
  offline shortcut on).
* For every scan that comes on screen the trainer draws the prediction
  through its own ``segment`` path, as the painter asks it to, so that time
  is spent here too (``--no-segmentation`` leaves it out).
* At each ``--milestones`` count of scans viewed, and at the end, the model
  the painter would be showing at that moment is recorded: the live weights
  while validation holds no lesion mark yet, otherwise the best checkpoint.
  ``score_session_replay.py`` then scores each of those on the held-out
  patients, which gives detection against scans viewed and minutes elapsed.

Session time is real time. Stretches in which the live trainer would sit
idle (before training starts, or while training or validation has no
annotation) are skipped instead of slept through, and loading the model is
not counted, because in a real session the trainer is already running.

One limit: the stored strokes corrected an earlier model's mistakes, not
this one's, so a replay approximates a corrective session. Timing a person at
the painter remains the real check.

A replay needs a fresh project (it refuses one that already has checkpoints)
made with ``prepare_offline_project.py --folds K --fold k --session-split``.
Outputs, all inside the project:
    replay/milestones.jsonl          one record per milestone (counts only)
    replay/snapshots/                live weights recorded during warm-up
    logs/replay_run_summary.json
    logs/offline_history.jsonl, offline_server_log.txt   as for offline runs
Create a ``STOP`` file in the project to abandon a replay.

    python run_session_replay.py --project D:\\RootPainterSync\\projects\\NAME_fold0 \\
        --gpu 0 --pace-seconds 20
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from mask_io import read_annotation
from run_offline_training import (build_offline_trainer, build_parser,
                                  git_commit, load_project, rollback_patience,
                                  training_config)

ROLES = ('train', 'val')


class SessionClock:
    """Session time: real time since the start plus idle time skipped."""

    def __init__(self, timer=time.monotonic):
        self.timer = timer
        self.started = timer()
        self.skipped = 0.0

    def now(self):
        return self.timer() - self.started + self.skipped

    def skip_to(self, seconds):
        """Jump forward to *seconds* of session time; never backwards."""
        gap = seconds - self.now()
        if gap > 0:
            self.skipped += gap


def parse_milestones(text):
    """``'25,50,100'`` -> ``[25, 50, 100]`` (sorted, distinct, positive)."""
    values = sorted({int(part) for part in str(text).split(',')
                     if part.strip()})
    if any(value < 1 for value in values):
        raise ValueError('Milestones are counts of scans viewed, from 1 up')
    return values


def session_order(settings):
    """The scans a user would work through: navigation order, held-out removed."""
    if 'test_file_names' not in settings:
        raise ValueError('Not a cross-validation fold project: it has no '
                         'held-out scans to score a replay on')
    held_out = set(settings['test_file_names'])
    return [name for name in settings['file_names'] if name not in held_out]


class SessionReplay:
    """Feeds a project's annotations to a trainer on a session schedule."""

    def __init__(self, project_dir, order, pace_seconds, warmup_scans=10,
                 milestones=(), tail_minutes=0.0, max_scans=None, clock=None):
        if pace_seconds <= 0:
            raise ValueError('The pace must be a positive number of seconds')
        self.project = Path(project_dir)
        self.full_order = list(order)
        self.order = self.full_order[:max_scans]
        self.pace = float(pace_seconds)
        self.warmup = min(int(warmup_scans), len(self.order))
        self.milestones = [count for count in sorted(set(milestones))
                           if count < len(self.order)]
        self.tail = float(tail_minutes) * 60
        self.clock = clock or SessionClock()
        self.pending_dir = self.project / 'replay' / 'pending'
        self.role_of = {}           # scan stem -> 'train' / 'val'
        self.has_lesion_mark = {}   # scan stem -> bool
        self.revealed = 0           # scans viewed so far
        self.epochs = 0
        self.segmented = 0
        self.segment_seconds = 0.0
        self.early_stop_scan = None

    def stage(self):
        """Hide every annotation; ``reveal_up_to`` brings them back in order."""
        if (self.project / 'replay').exists():
            raise ValueError('This project already holds a replay; use a '
                             'fresh project')
        for role in ROLES:
            target = self.pending_dir / role
            target.mkdir(parents=True)
            for path in sorted(
                    (self.project / 'annotations' / role).glob('*.png')):
                self.role_of[path.stem] = role
                self.has_lesion_mark[path.stem] = bool(
                    read_annotation(path)[0].any())
                os.replace(path, target / path.name)
        strays = set(self.role_of) - {Path(name).stem
                                      for name in self.full_order}
        if strays:
            raise ValueError(f'{len(strays)} annotation file(s) belong to no '
                             'scan of the session')

    def arrived(self, seconds):
        """Scans finished *seconds* into the session."""
        return min(len(self.order), int((seconds + 1e-9) // self.pace))

    def reveal_up_to(self, count):
        """Put back the annotations of the first *count* scans."""
        for name in self.order[self.revealed:count]:
            stem = Path(name).stem
            role = self.role_of.get(stem)
            if role:
                os.replace(
                    self.pending_dir / role / (stem + '.png'),
                    self.project / 'annotations' / role / (stem + '.png'))
        self.revealed = max(self.revealed, count)

    def counts(self):
        """Annotations among the scans viewed so far, per side."""
        viewed = [Path(name).stem for name in self.order[:self.revealed]]
        result = {}
        for role in ROLES:
            stems = [stem for stem in viewed if self.role_of.get(stem) == role]
            result[f'{role}_annotations'] = len(stems)
            result[f'{role}_with_lesion_marks'] = sum(
                self.has_lesion_mark[stem] for stem in stems)
        return result

    def can_train(self):
        """The trainer only runs an epoch with both sides annotated."""
        counts = self.counts()
        return all(counts[f'{role}_annotations'] for role in ROLES)

    def run(self, trainer, snapshot, segment=None):
        """Run the session; returns False if a STOP file abandoned it.

        *snapshot(label)* records the model on screen at a milestone, and
        *segment(names)* draws predictions for scans as they come on screen.
        """
        clock, total = self.clock, len(self.order)
        end = total * self.pace + self.tail
        # Nothing trains during the first reviews, so nothing is waited for.
        clock.skip_to(self.warmup * self.pace)
        on_screen = self.warmup
        due = list(self.milestones)
        while True:
            now = clock.now()
            self.reveal_up_to(self.arrived(now))
            while due and due[0] <= self.revealed:
                snapshot(due.pop(0))
            if getattr(trainer, 'stop_reason', None) == 'stop_file':
                return False
            if segment is not None:
                # Scan number ``revealed`` (from 0) is the one on screen now.
                names = self.order[on_screen:self.revealed + 1]
                if names:
                    started = time.monotonic()
                    segment(names)
                    self.segment_seconds += time.monotonic() - started
                    self.segmented += len(names)
                on_screen = max(on_screen, min(self.revealed + 1, total))
            if self.revealed >= total and now >= end:
                break
            if trainer.training and self.can_train():
                trainer.epoch += 1
                trainer.train_one_epoch()
                self.epochs += 1
                if not trainer.training and self.early_stop_scan is None:
                    self.early_stop_scan = self.arrived(clock.now())
            else:
                # The live trainer would be idle until the next scan is done.
                clock.skip_to((self.revealed + 1) * self.pace
                              if self.revealed < total else end)
        snapshot('end')
        return True


def parse_args(argv=None):
    parser = build_parser(description=__doc__)
    parser.add_argument('--pace-seconds', type=float, default=20.0,
                        help='time the user spends on each scan (default 20)')
    parser.add_argument('--milestones', default='25,50,100,200',
                        help='scans viewed at which the model on screen is '
                             'recorded; the end of the session always is')
    parser.add_argument('--warmup-scans', type=int, default=None,
                        help='scans reviewed before training starts (default: '
                             "the project's auto_start_training_after_reviews)")
    parser.add_argument('--max-scans', type=int, default=None,
                        help='end the session after this many scans')
    parser.add_argument('--tail-minutes', type=float, default=0.0,
                        help='keep training this long after the last scan')
    parser.add_argument('--validation-cache', action='store_true',
                        help='score each best checkpoint once instead of '
                             'every epoch (faster than the live trainer)')
    parser.add_argument('--no-segmentation', action='store_true',
                        help='do not draw the prediction for each scan that '
                             'comes on screen')
    args = parser.parse_args(argv)
    if args.trainer_commit or args.max_minutes is not None:
        parser.error('--trainer-commit and --max-minutes do not apply to a '
                     'session replay')
    args.no_validation_cache = not args.validation_cache
    args.milestones = parse_milestones(args.milestones)
    return args


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
    config = training_config(project_dir, images_dir, args.model_type, seed)
    if any((project_dir / 'models').glob('*.pkl')):
        raise SystemExit('This project already has checkpoints; a replay '
                         'needs a fresh project')
    warmup = (args.warmup_scans if args.warmup_scans is not None
              else settings.get('auto_start_training_after_reviews', 10))
    try:
        replay = SessionReplay(
            project_dir, session_order(settings), args.pace_seconds,
            warmup_scans=warmup, milestones=args.milestones,
            tail_minutes=args.tail_minutes, max_scans=args.max_scans)
        replay.stage()
    except ValueError as error:
        raise SystemExit(str(error)) from error

    trainer = build_offline_trainer(project_dir, sync_dir, args)
    if trainer.stop_path.exists():
        raise SystemExit(f'Remove {trainer.stop_path} before starting')

    import torch  # pylint: disable=import-outside-toplevel
    import model_utils  # pylint: disable=import-outside-toplevel

    patience = rollback_patience(args)
    rollback = 'off' if patience is None else patience
    print(f'Session replay: project={project_dir.name} '
          f'model={args.model_type} seed={seed} '
          f'gpu={os.environ.get("CUDA_VISIBLE_DEVICES", "all")} '
          f'scans={len(replay.order)} annotated={len(replay.role_of)} '
          f'pace={args.pace_seconds}s warmup={replay.warmup} '
          f'rollback={rollback}', flush=True)
    started = time.time()
    # The live trainer is already running, with its first model made, when
    # the user reaches the scan that starts training.
    trainer.start_training(config)
    if args.foreground_tile_fraction is not None:
        trainer.train_set.foreground_tile_fraction = (
            args.foreground_tile_fraction)

    snapshot_dir = project_dir / 'replay' / 'snapshots'
    milestones_path = project_dir / 'replay' / 'milestones.jsonl'
    segment_config = {'dataset_dir': config['dataset_dir'],
                      'seg_dir': config['seg_dir'],
                      'model_dir': config['model_dir'],
                      'model_type': args.model_type}
    replay.clock = clock = SessionClock()

    def snapshot(label):
        """Record the model the painter would be showing right now."""
        # pylint: disable=protected-access
        live = trainer._get_provisional_live_model_for_segment(segment_config)
        if live is not None and replay.epochs:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            name = (f'scans_{label:04d}' if isinstance(label, int)
                    else str(label))
            path = snapshot_dir / f'{name}_live.pkl'
            torch.save(live.state_dict(), path)
            shown = 'live weights'
        else:
            path = Path(model_utils.get_latest_model_paths(
                config['model_dir'], 1)[0])
            shown = 'checkpoint'
        record = dict(
            milestone=label, scans_viewed=replay.revealed,
            session_minutes=round(clock.now() / 60, 2),
            epochs=replay.epochs, ui_model=shown,
            model=path.relative_to(project_dir).as_posix(),
            validation_has_lesion_mark=bool(trainer.validation_has_foreground),
            **replay.counts())
        with open(milestones_path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(f'[replay] milestone {label}: {record["scans_viewed"]} scans '
              f'viewed, {record["session_minutes"]} min, showing {shown}',
              flush=True)

    def segment(names):
        # The trainer prints each file name; scan names stay out of the log.
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                trainer.segment(dict(segment_config, file_names=list(names)))

    train_one_epoch = trainer.train_one_epoch

    def train_and_report():
        train_one_epoch()
        counts = replay.counts()
        print(f'[replay] epoch {trainer.epoch} done; '
              f'{replay.revealed} of {len(replay.order)} scans viewed; '
              f'annotations train {counts["train_annotations"]} '
              f'val {counts["val_annotations"]}; '
              f'session {clock.now() / 60:.1f} min', flush=True)

    trainer.train_one_epoch = train_and_report
    completed = replay.run(
        trainer, snapshot, None if args.no_segmentation else segment)

    best = model_utils.get_latest_model_paths(config['model_dir'], 1)[0]
    summary = {
        'project': project_dir.name,
        'model_type': args.model_type,
        'loss_type': args.loss_type,
        'foreground_tile_fraction': args.foreground_tile_fraction,
        'training_seed': seed,
        'split_method': settings.get('split_method'),
        'completed': completed,
        'pace_seconds': args.pace_seconds,
        'warmup_scans': replay.warmup,
        'tail_minutes': args.tail_minutes,
        'session_scans': len(replay.order),
        'scans_viewed': replay.revealed,
        'milestones': replay.milestones,
        'session_minutes': round(clock.now() / 60, 2),
        'idle_minutes_skipped': round(clock.skipped / 60, 2),
        'epochs_run': replay.epochs,
        'scans_segmented': replay.segmented,
        'segmentation_minutes': round(replay.segment_seconds / 60, 2),
        'early_stopping_fired_at_scan': replay.early_stop_scan,
        'candidate_rollback': rollback,
        'rollbacks': trainer.rollbacks,
        'best_validation_cache': 'on' if args.validation_cache else 'off',
        'final_checkpoint': os.path.basename(best),
        'batch_size': trainer.bs,
        'workers': trainer.num_workers,
        'duration_minutes': round((time.time() - started) / 60, 2),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'torch': torch.__version__,
        'devices': [torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())],
        **replay.counts(),
    }
    with open(project_dir / 'logs' / 'replay_run_summary.json', 'w',
              encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    if not completed:
        raise SystemExit('Replay abandoned by STOP file')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
