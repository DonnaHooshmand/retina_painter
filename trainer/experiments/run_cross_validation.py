"""Prepare, train and score a patient-level cross-validation run.

One command per configuration. It builds one project per fold
(``<name>_fold0`` ...), trains them on the GPUs given (one fold per GPU at a
time), and scores the held-out patients. Everything after ``--`` is passed to
the trainer, so a configuration is its preparation options plus its training
options. There are two kinds of run.

**All annotations at once** (the end of a session), trained with
``run_offline_training.py`` for ``--max-minutes`` and scored with
``score_cross_validation.py``:

    python run_cross_validation.py --dataset <curated dataset> \\
        --sync-dir D:\\RootPainterSync --name cv_sparse_retfound_rfa \\
        --model-type retfound_rfa --folds 4 --gpus 0,1 --max-minutes 45 \\
        -- --loss-type combined

**A replayed session** (``--session-pace SECONDS``): the annotations arrive
one scan at a time while the trainer runs, as they do for a user, and the
model on screen is scored at milestones of the session. Trained with
``run_session_replay.py`` and scored with ``score_session_replay.py``;
``--max-minutes`` does not apply, the session lasts as long as its scans:

    python run_cross_validation.py --dataset <curated dataset> \\
        --sync-dir D:\\RootPainterSync --name replay_retfound_rfa \\
        --model-type retfound_rfa --folds 4 --gpus 0,1 --session-pace 20 \\
        -- --milestones 25,50,100,200

Re-running the same command resumes: folds that already have a run summary
are not trained again, and stored scores are reused. (A replayed fold that
stopped half way cannot be resumed; delete its project folder first.)
Progress is appended to ``<sync>/projects/<name>_cross_validation.log`` or
``<name>_session_replay.log``; only counts are printed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from prepare_offline_project import (MODEL_TYPES, SUPERVISION_POLICIES,
                                     prepare_offline_project)

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    train_args = []
    if '--' in argv:
        cut = argv.index('--')
        argv, train_args = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--sync-dir', required=True, type=Path)
    parser.add_argument('--name', required=True, help='run name')
    parser.add_argument('--model-type', default='retfound_rfa',
                        choices=MODEL_TYPES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--supervision', default='sparse',
                        choices=SUPERVISION_POLICIES)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--gpus', default='0',
                        help='comma-separated GPU indices, one fold each')
    parser.add_argument('--max-minutes', type=float, default=45,
                        help='training budget per fold (not for a replayed '
                             'session)')
    parser.add_argument('--session-pace', type=float, default=None,
                        help='replay a session instead: seconds the user '
                             'spends on each scan')
    parser.add_argument('--baseline', default=None,
                        help='another run name to compare against when scoring')
    args = parser.parse_args(argv)
    args.train_args = train_args
    return args


def fold_name(name, fold):
    return f'{name}_fold{fold}'


def training_command(project, gpu, max_minutes, train_args):
    return [sys.executable, '-u', str(HERE / 'run_offline_training.py'),
            '--project', str(project), '--gpu', str(gpu),
            '--max-minutes', str(max_minutes), *train_args]


def replay_commands(project, gpu, pace, train_args, sync_dir, name):
    """Replay one fold's session, then score its milestones on the same GPU."""
    return [
        [sys.executable, '-u', str(HERE / 'run_session_replay.py'),
         '--project', str(project), '--gpu', str(gpu),
         '--pace-seconds', str(pace), *train_args],
        [sys.executable, str(HERE / 'score_session_replay.py'),
         '--sync-dir', str(sync_dir), '--name', name,
         '--project', str(project), '--gpu', str(gpu)],
    ]


def main(argv=None):
    args = parse_args(argv)
    replay = args.session_pace is not None
    projects_dir = args.sync_dir.resolve() / 'projects'
    kind = 'session_replay' if replay else 'cross_validation'
    log_path = projects_dir / f'{args.name}_{kind}.log'
    summary_name = ('replay_run_summary.json' if replay
                    else 'offline_run_summary.json')

    def log(message):
        line = f'{datetime.now().isoformat(timespec="seconds")} {message}'
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as stream:
            stream.write(line + '\n')

    pending = []
    for fold in range(args.folds):
        project = projects_dir / fold_name(args.name, fold)
        if not project.exists():
            summary = prepare_offline_project(
                args.dataset, args.sync_dir, project.name,
                model_type=args.model_type, seed=args.seed,
                supervision=args.supervision, folds=args.folds, fold=fold,
                session_split=replay)
            log(f'fold {fold}: prepared, images per role '
                f'{summary["split_images"]}')
        if (project / 'logs' / summary_name).is_file():
            log(f'fold {fold}: already trained')
            if replay:
                # Its milestones may still need scoring.
                pending.append((fold, project, True))
        else:
            pending.append((fold, project, False))

    def commands_for(project, gpu, trained):
        if replay:
            commands = replay_commands(
                project, gpu, args.session_pace, args.train_args,
                args.sync_dir, args.name)
            return commands[1:] if trained else commands
        return [training_command(project, gpu, args.max_minutes,
                                 args.train_args)]

    free = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
    running = {}
    failed = []

    def start(gpu, job):
        """Start the next command of *job* on *gpu*."""
        command = job['commands'].pop(0)
        job['process'] = subprocess.Popen(
            command, cwd=HERE, stdout=job['stdout'], stderr=job['stderr'])
        running[gpu] = job

    while pending or running:
        while pending and free:
            fold, project, trained = pending.pop(0)
            gpu = free.pop(0)
            job = dict(
                fold=fold, project=project,
                commands=commands_for(project, gpu, trained),
                stdout=open(project / 'logs' / 'offline_stdout.log', 'a',
                            encoding='utf-8'),
                stderr=open(project / 'logs' / 'offline_stderr.log', 'a',
                            encoding='utf-8'))
            start(gpu, job)
            log(f'fold {fold}: started on GPU {gpu}')
        time.sleep(15)
        for gpu, job in list(running.items()):
            code = job['process'].poll()
            if code is None:
                continue
            fold, project = job['fold'], job['project']
            succeeded = (code == 0
                         and (project / 'logs' / summary_name).is_file())
            if succeeded and job['commands']:
                log(f'fold {fold}: training finished, scoring its milestones')
                start(gpu, job)
                continue
            job['stdout'].close()
            job['stderr'].close()
            del running[gpu]
            free.append(gpu)
            if succeeded:
                log(f'fold {fold}: finished')
            else:
                failed.append(fold)
                log(f'fold {fold}: FAILED (exit code {code}); see its '
                    'logs/offline_stderr.log')
    if failed:
        raise SystemExit(f'Folds failed: {failed}')

    scorer = ('score_session_replay.py' if replay
              else 'score_cross_validation.py')
    command = [sys.executable, str(HERE / scorer),
               '--sync-dir', str(args.sync_dir), '--name', args.name,
               '--gpu', args.gpus.split(',')[0].strip()]
    if args.baseline:
        command += ['--baseline', args.baseline]
    log('scoring held-out patients')
    scored = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    for line in (scored.stdout + scored.stderr).splitlines():
        log(line)
    if scored.returncode != 0:
        raise SystemExit('Scoring failed')
    log('done')


if __name__ == '__main__':
    main()
