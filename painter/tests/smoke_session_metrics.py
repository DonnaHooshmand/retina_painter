"""Qt checks for the record of a session's time and painted pixels.

RootPainter's interaction log (``logs/client.csv``) and its metrics export
are still part of the painter. These checks drive the real painter window
offscreen on synthetic scans and confirm that

* the log records each scan coming on screen, each stroke made through the
  scene's own mouse handlers and each save;
* the metrics export runs on a project whose training started automatically
  (that log line names no file and used to stop the export);
* a scan reviewed as absent in a project that counts absent scans as clean
  scans is exported with the strokes the user painted, not with the whole
  scan, and the time each scan was on screen is exported too;
* a project without that option is exported as before;
* ``annotation_effort.py`` reports the same pixels in the order viewed.

Run directly with the painter environment (PyQt5, no pytest needed):

    python smoke_session_metrics.py
"""

import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent))

from smoke_annotation_session import NAMES, QtCore  # noqa: E402
from smoke_absent_background import (  # noqa: E402
    PIXELS, _close, _next, _on_screen, _session, patched_painter)
import annotation_effort  # noqa: E402
import plot_seg_metrics  # noqa: E402


class _At:
    """A mouse event at a scene position, as the scene's handlers read it."""

    def __init__(self, x, y):
        self._position = QtCore.QPointF(x, y)

    def scenePos(self):
        return self._position


def _stroke(window, x, y):
    """One brush stroke through the scene's own press and release handlers."""
    window.scene.mousePressEvent(_At(x, y))
    window.scene.mouseReleaseEvent(_At(x, y))


def _review_three_scans(window, root):
    """Scan 1 absent with a background stroke, scan 2 absent and unpainted,
    scan 3 present with a lesion stroke. Returns the pixels painted."""
    window.set_background_color(False)
    _stroke(window, 100, 400)
    _lesion, background = _on_screen(window, root)
    assert _lesion == 0 and 0 < background < PIXELS
    window.review_widget.set_review('absent', True)
    _next(window)

    assert Path(window.image_path).name == NAMES[1]
    window.review_widget.set_review('absent', True)
    _next(window)

    assert Path(window.image_path).name == NAMES[2]
    window.set_foreground_color(False)
    _stroke(window, 285, 216)
    lesion, _background = _on_screen(window, root)
    assert lesion > 0 and _background == 0
    window.review_widget.set_review('present', True)
    _next(window)
    assert Path(window.image_path).name == NAMES[3]
    # What the painter logs when the tenth review starts training.
    window.log('automatic_start_training,completed_reviews:10')
    window.log_debounced()
    return background, lesion


def _export(project, root):
    """Run the metrics export as Extras > Export metrics CSV does."""
    csv_path = root / 'exported_metrics.csv'
    plot_seg_metrics.Thread(str(project), str(csv_path), NAMES).run()
    with open(csv_path, newline='') as stream:
        return {row['file_name']: row
                for row in csv.DictReader(stream, quotechar='|')}


def check_the_log_records_scans_strokes_and_saves(root, dialogs):
    _sync, project, window = _session(root)
    try:
        _review_three_scans(window, root)
    finally:
        _close(window)
    lines = (project / 'logs' / 'client.csv').read_text().splitlines()
    logged = [(line.split(',')[2],
               next((part[len('fname:'):] for part in line.split(',')
                     if part.startswith('fname:')), None))
              for line in lines]
    for index in range(4):
        assert ('update_file_end', NAMES[index]) in logged
    for index in (0, 2):
        assert ('mouse_press', NAMES[index]) in logged
        assert ('mouse_release', NAMES[index]) in logged
    for index in range(3):
        assert ('save_annotation', NAMES[index]) in logged
    assert ('automatic_start_training', None) in logged
    # Every line carries a clock time the effort figures can use.
    assert all(float(line.split(',')[1]) > 0 for line in lines)
    assert not dialogs


def check_the_export_reports_strokes_painted_and_time_on_screen(
        root, dialogs):
    _sync, project, window = _session(root)
    try:
        background, lesion = _review_three_scans(window, root)
    finally:
        _close(window)

    rows = _export(project, root)

    painted = rows[NAMES[0]]
    assert (int(painted['annot_fg']), int(painted['annot_bg'])) == (
        0, background)
    assert painted['strokes_source'] == 'review_snapshot'
    assert int(painted['clicks']) == 1
    unpainted = rows[NAMES[1]]
    assert (int(unpainted['annot_fg']), int(unpainted['annot_bg'])) == (0, 0)
    assert unpainted['strokes_source'] == 'review_snapshot'
    assert int(unpainted['clicks']) == 0
    present = rows[NAMES[2]]
    assert (int(present['annot_fg']), int(present['annot_bg'])) == (lesion, 0)
    assert present['strokes_source'] == 'annotation'
    for row in rows.values():
        assert float(row['screen_time_s']) >= 0
        assert float(row['annot_duration_s']) >= 0
    # The corrected segmentation of a clean scan is still empty.
    assert int(painted['area_true']) == 0

    effort = annotation_effort.effort_rows(project)
    assert [row['file_name'] for row in effort] == NAMES[:4]
    assert [row['painted_fg'] + row['painted_bg'] for row in effort] == [
        background, 0, lesion, 0]
    assert effort[-1]['cumulative_painted_pixels'] == background + lesion
    assert not dialogs


def check_projects_without_the_option_are_exported_as_before(root, dialogs):
    _sync, project, window = _session(root, enabled=False)
    try:
        background, lesion = _review_three_scans(window, root)
    finally:
        _close(window)

    rows = _export(project, root)

    painted = rows[NAMES[0]]
    assert (int(painted['annot_fg']), int(painted['annot_bg'])) == (
        0, background)
    assert painted['strokes_source'] == 'annotation'
    # Nothing was painted on the second scan, so it has no annotation file.
    assert rows[NAMES[1]]['strokes_source'] == 'none'
    assert int(rows[NAMES[2]]['annot_fg']) == lesion
    assert not dialogs


CHECKS = [
    check_the_log_records_scans_strokes_and_saves,
    check_the_export_reports_strokes_painted_and_time_on_screen,
    check_projects_without_the_option_are_exported_as_before,
]


def main():
    for check in CHECKS:
        with patched_painter() as dialogs, \
                tempfile.TemporaryDirectory() as directory:
            check(Path(directory), dialogs)
        print('ok  ', check.__name__)
    print(f'{len(CHECKS)} checks passed')


if __name__ == '__main__':
    main()
