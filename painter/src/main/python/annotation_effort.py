"""Annotation effort of a session: time spent and pixels painted per scan.

RootPainter's paper shows how much the user had to paint as a session went on
(cumulative annotated pixels against annotation time). This module computes
the same quantities for a RetinaPainter project from what the painter already
stores:

* ``logs/client.csv``: every mouse press and release, every scan coming on
  screen and every save, with timestamps;
* the annotation files, and the review snapshots that keep the user's own
  strokes for a scan reviewed as absent in a project that counts absent scans
  as clean scans (there the annotation file marks the whole scan as
  background, which is not what the user painted);
* ``reviews/review_records.json``: the decision for each scan.

It needs no Qt. Run it on one project, or on two to compare sessions:

    python annotation_effort.py --project <project folder>
    python annotation_effort.py --project <A> --project <B> \\
        --label RetinaPainter --label U-Net --out effort.png

Each project gets ``metrics/annotation_effort.csv`` (one row per scan viewed,
in the order viewed). Only totals are printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

from interaction_time import (events_from_client_log, get_annot_duration_s,
                              screen_time_per_fname_s)
from review_utils import (absent_review_is_complete, is_whole_scan_background,
                          load_review_document)

DEFAULT_PAUSE_S = 120.0
# Where the pixel counts of a scan came from.
STROKES_FROM_ANNOTATION = 'annotation'
STROKES_FROM_REVIEW_SNAPSHOT = 'review_snapshot'
NO_STROKES = 'none'

CSV_COLUMNS = (
    'viewing_order', 'file_name', 'project_order', 'session_s',
    'screen_time_s', 'visits', 'paused_s', 'annot_duration_s', 'clicks',
    'painted_fg', 'painted_bg', 'strokes_source', 'decision',
    'review_complete', 'side', 'cumulative_screen_time_s',
    'cumulative_annot_duration_s', 'cumulative_painted_pixels')


def project_settings(project_dir) -> dict:
    """The contents of the project's ``.seg_proj`` file."""
    files = sorted(Path(project_dir).glob('*.seg_proj'))
    if len(files) != 1:
        raise ValueError(
            f'Expected one .seg_proj file in {project_dir}, found '
            f'{len(files)}')
    with open(files[0], encoding='utf-8') as stream:
        return json.load(stream)


def annotation_path(project_dir, image_fname):
    """``(path, side)`` of a scan's annotation file, or ``(None, None)``."""
    png_name = Path(image_fname).stem + '.png'
    for side in ('train', 'val'):
        path = Path(project_dir) / 'annotations' / side / png_name
        if path.is_file():
            return path, side
    return None, None


def stroke_pixel_counts(strokes_path):
    """``(foreground, background)`` pixels painted in a strokes file.

    Decoded as the trainer decodes annotations: red is foreground and green is
    background, on pixels that are not fully transparent.
    """
    if strokes_path is None or not Path(strokes_path).is_file():
        return 0, 0
    with Image.open(strokes_path) as image:
        pixels = np.asarray(image.convert('RGBA'))
    visible = pixels[:, :, 3] > 0
    foreground = np.logical_and(pixels[:, :, 0] > 0, visible)
    background = np.logical_and(pixels[:, :, 1] > 0, visible)
    return int(np.count_nonzero(foreground)), int(np.count_nonzero(background))


def painted_strokes(project_dir, image_fname, review,
                    absent_scans_as_background):
    """``(path, source)`` of the file holding what the user painted on a scan.

    The rule is the painter's (``RetinaPainter.strokes_path_for_display``).
    Normally the strokes are the annotation file. For a completed absent
    review in a project that counts absent scans as clean scans the annotation
    file marks the whole scan as background; the user's own strokes are then
    the ones snapshotted with the review, or none (``path`` is ``None``).
    """
    path, _side = annotation_path(project_dir, image_fname)
    if path is None:
        return None, NO_STROKES
    if not (absent_scans_as_background
            and absent_review_is_complete(review)
            and is_whole_scan_background(path)):
        return path, STROKES_FROM_ANNOTATION
    snapshot = review.get('annotation_snapshot')
    if snapshot and (Path(project_dir) / snapshot).is_file():
        return Path(project_dir) / snapshot, STROKES_FROM_REVIEW_SNAPSHOT
    return None, STROKES_FROM_REVIEW_SNAPSHOT


def load_events(project_dir):
    """Events of the project's client log, or an empty list without one."""
    log_path = Path(project_dir) / 'logs' / 'client.csv'
    if not log_path.is_file():
        return []
    return events_from_client_log(log_path)


def effort_rows(project_dir, pause_s=DEFAULT_PAUSE_S):
    """One row per scan the user viewed, in the order first viewed.

    A scan counts as viewed when the log shows it on screen, or when it has a
    review or an annotation (projects without a log). Scans never opened are
    left out. Without a log the order is the project's own and every time is
    zero.
    """
    project_dir = Path(project_dir)
    settings = project_settings(project_dir)
    absent_as_background = bool(
        settings.get('absent_scans_as_background', False))
    reviews = load_review_document(project_dir)['reviews']
    events = load_events(project_dir)
    screen = screen_time_per_fname_s(events, pause_s)
    first_seen = {}
    for event in sorted(events, key=lambda e: e.time):
        if event.name == 'update_file_end':
            first_seen.setdefault(event.fname, event.time)
    session_start = min(first_seen.values()) if first_seen else None

    rows = []
    for index, fname in enumerate(settings['file_names']):
        stem = Path(fname).stem
        review = reviews.get(fname)
        strokes, source = painted_strokes(
            project_dir, fname, review, absent_as_background)
        _path, side = annotation_path(project_dir, fname)
        seconds, visits, paused = screen.get(stem, (0.0, 0, 0.0))
        if not (visits or review is not None or side is not None):
            continue
        painted_fg, painted_bg = stroke_pixel_counts(strokes)
        duration, clicks = (get_annot_duration_s(events, fname)
                            if events else (0, 0))
        rows.append({
            'file_name': fname,
            'project_order': index + 1,
            'session_s': (None if stem not in first_seen
                          else first_seen[stem] - session_start),
            'screen_time_s': seconds,
            'visits': visits,
            'paused_s': paused,
            'annot_duration_s': duration,
            'clicks': clicks,
            'painted_fg': painted_fg,
            'painted_bg': painted_bg,
            'strokes_source': source,
            'decision': (review or {}).get('decision'),
            'review_complete': bool((review or {}).get('review_complete')),
            'side': side,
        })
    rows.sort(key=lambda row: (row['session_s'] is None,
                               row['session_s'] or 0.0,
                               row['project_order']))
    totals = dict(screen=0.0, annot=0.0, pixels=0)
    for order, row in enumerate(rows, start=1):
        totals['screen'] += row['screen_time_s']
        totals['annot'] += row['annot_duration_s']
        totals['pixels'] += row['painted_fg'] + row['painted_bg']
        row['viewing_order'] = order
        row['cumulative_screen_time_s'] = totals['screen']
        row['cumulative_annot_duration_s'] = totals['annot']
        row['cumulative_painted_pixels'] = totals['pixels']
    return rows


def summarise(rows) -> dict:
    """Totals for a session; contains no file name."""
    screen = [row['screen_time_s'] for row in rows if row['visits']]
    painted = [row['painted_fg'] + row['painted_bg'] for row in rows]
    decisions = {}
    for row in rows:
        if row['review_complete']:
            decisions[row['decision']] = decisions.get(row['decision'], 0) + 1
    return {
        'scans_viewed': len(rows),
        'scans_painted': sum(pixels > 0 for pixels in painted),
        'scans_with_lesion_strokes': sum(
            row['painted_fg'] > 0 for row in rows),
        'completed_reviews': decisions,
        'session_minutes': sum(screen) / 60,
        'paused_minutes': sum(row['paused_s'] for row in rows) / 60,
        'seconds_per_scan_mean': statistics.mean(screen) if screen else 0.0,
        'seconds_per_scan_median': (statistics.median(screen)
                                    if screen else 0.0),
        'annotation_minutes': sum(
            row['annot_duration_s'] for row in rows) / 60,
        'clicks': sum(row['clicks'] for row in rows),
        'painted_pixels_foreground': sum(row['painted_fg'] for row in rows),
        'painted_pixels_background': sum(row['painted_bg'] for row in rows),
        'scans_counted_as_clean': sum(
            row['strokes_source'] == STROKES_FROM_REVIEW_SNAPSHOT
            for row in rows),
    }


def write_effort_csv(rows, csv_path):
    """Write the rows of a session; returns the path written."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in CSV_COLUMNS})
    return csv_path


def plot_effort(sessions, out_path):
    """Draw cumulative painted pixels against time for one or more sessions.

    ``sessions`` is a list of ``(label, rows)``. The left panel is
    RootPainter's figure: pixels painted so far against time spent painting.
    The right panel uses the time the scans were on screen, which includes
    scans that needed no correction.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (by_annotation, by_session) = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True)
    for label, rows in sessions:
        pixels = [0] + [row['cumulative_painted_pixels'] for row in rows]
        annotation = [0.0] + [row['cumulative_annot_duration_s'] / 60
                              for row in rows]
        session = [0.0] + [row['cumulative_screen_time_s'] / 60
                           for row in rows]
        by_annotation.plot(annotation, pixels, label=label)
        by_session.plot(session, pixels, label=label)
    by_annotation.set_xlabel('Time spent painting (minutes)')
    by_annotation.set_ylabel('Pixels painted so far')
    by_session.set_xlabel('Session time (minutes)')
    for axes in (by_annotation, by_session):
        axes.grid(True, alpha=0.3)
    if len(sessions) > 1:
        by_annotation.legend()
    figure.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project', action='append', required=True,
                        type=Path, help='project folder (repeatable)')
    parser.add_argument('--label', action='append', default=[],
                        help='name of each project in the figure')
    parser.add_argument('--out', type=Path, default=None,
                        help='figure file (default: metrics/'
                             'annotation_effort.png in the first project)')
    parser.add_argument('--pause-seconds', type=float,
                        default=DEFAULT_PAUSE_S,
                        help='a longer gap between logged events counts as '
                             'time away, not session time (default 120)')
    args = parser.parse_args(argv)
    if args.label and len(args.label) != len(args.project):
        parser.error('give one --label per --project, or none')
    labels = args.label or [project.name for project in args.project]

    sessions = []
    for label, project in zip(labels, args.project):
        rows = effort_rows(project, args.pause_seconds)
        csv_path = write_effort_csv(
            rows, project / 'metrics' / 'annotation_effort.csv')
        sessions.append((label, rows))
        print(f'{label}: {csv_path}')
        for key, value in summarise(rows).items():
            shown = f'{value:.1f}' if isinstance(value, float) else value
            print(f'  {key}: {shown}')
    out_path = args.out or (args.project[0] / 'metrics'
                            / 'annotation_effort.png')
    print(f'figure: {plot_effort(sessions, out_path)}')


if __name__ == '__main__':
    main()
