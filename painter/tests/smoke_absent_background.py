"""Qt checks for "scans marked RIPL absent are clean scans" (synthetic data).

A project made with that option trains on a scan reviewed as absent as
background everywhere. These checks drive the real painter window offscreen
and confirm that

* completing a review as absent writes a whole-scan background annotation on
  the scan's fixed side, with or without strokes;
* returning to the scan shows the strokes the user painted, not a green wash,
  and saving again rewrites nothing;
* changing the decision puts the user's strokes back, or removes the file when
  there are none;
* red marks on an absent scan still block the review and are never replaced;
* a project without the option keeps sparse strokes only;
* the state survives closing and reopening the project.

Run directly with the painter environment (PyQt5, no pytest needed):

    python smoke_absent_background.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent))

from smoke_annotation_session import (  # noqa: E402
    NAMES, QtGui, _create, _open, _paint, _paths)
from smoke_review_project import mask_io, patched_painter  # noqa: E402
from review_utils import load_review  # noqa: E402

PIXELS = 600 * 600
GREEN_STROKE = 80 * 12
RED_STROKE = 30 * 12


def _session(root, enabled=True):
    """Open the synthetic 12-scan project; scans 5 and 10 are validation."""
    sync = root / 'sync'
    _create(sync)
    project, config_path, _ = _paths(sync)
    if enabled:
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['absent_scans_as_background'] = True
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    return sync, project, _open(sync)


def _close(window):
    window.close()
    window.log_debounce.stop()


def _paint_green(window):
    painter = QtGui.QPainter(window.scene.annot_pixmap)
    painter.fillRect(80, 400, 80, 12, QtGui.QColor(0, 255, 0, 180))
    painter.end()
    window.scene.annot_pixmap_holder.setPixmap(window.scene.annot_pixmap)


def _paint_red(window):
    painter = QtGui.QPainter(window.scene.annot_pixmap)
    painter.fillRect(270, 210, 30, 12, QtGui.QColor(255, 0, 0, 180))
    painter.end()
    window.scene.annot_pixmap_holder.setPixmap(window.scene.annot_pixmap)


def _counts(path):
    """(foreground, background) pixel counts, decoded as the trainer does."""
    foreground, background = mask_io.read_annotation(path)
    return int(foreground.sum()), int(background.sum())


def _on_screen(window, root):
    """(foreground, background) pixel counts of the strokes in the scene."""
    path = root / 'strokes_on_screen.png'
    window.scene.annot_pixmap.save(str(path), 'PNG')
    return _counts(path)


def _annotation(project, name):
    """(side, path) of a scan's annotation file, or (None, None)."""
    for side in ('train', 'val'):
        path = project / 'annotations' / side / name
        if path.is_file():
            return side, path
    return None, None


def _decide(window, decision):
    """Choose a decision as a user does (this marks the review as edited)."""
    combo = window.review_widget.decision_combo
    combo.setCurrentIndex(combo.findData(decision))
    window.review_widget.complete_checkbox.setChecked(True)


def _next(window):
    window.nav.next_image_button.click()


def _previous(window):
    window.nav.prev_image_button.click()


def check_absent_review_makes_the_whole_scan_background(root, dialogs):
    _sync, project, window = _session(root)
    try:
        # Scan 1 (training side): nothing painted, decided absent.
        window.review_widget.set_review('absent', True)
        _next(window)
        assert Path(window.image_path).name == NAMES[1]
        side, path = _annotation(project, NAMES[0])
        assert side == 'train'
        assert _counts(path) == (0, PIXELS)
        record = load_review(project, NAMES[0])
        assert record['decision'] == 'absent' and record['review_complete']
        assert record['annotation_snapshot'] is None

        # Uncertain scans are still not trained on.
        for index in (1, 2, 3):
            window.review_widget.set_review('uncertain', True)
            _next(window)
            assert _annotation(project, NAMES[index]) == (None, None)

        # Scan 5 is a validation scan: its file goes to the validation side.
        assert Path(window.image_path).name == NAMES[4]
        _paint_green(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        side, path = _annotation(project, NAMES[4])
        assert side == 'val'
        assert _counts(path) == (0, PIXELS)

        # A present scan keeps exactly what was painted on it.
        _paint(window)
        window.review_widget.set_review('present', True)
        _next(window)
        side, path = _annotation(project, NAMES[5])
        assert side == 'train'
        assert _counts(path) == (RED_STROKE, GREEN_STROKE)
        assert not dialogs
    finally:
        _close(window)


def check_returning_shows_own_strokes_and_rewrites_nothing(root, dialogs):
    _sync, project, window = _session(root)
    try:
        _paint_green(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        _side, path = _annotation(project, NAMES[0])
        assert _counts(path) == (0, PIXELS)
        record = load_review(project, NAMES[0])
        # The review keeps what the user painted.
        assert _counts(project / record['annotation_snapshot']) == (
            0, GREEN_STROKE)
        written = path.stat().st_mtime_ns

        _previous(window)
        assert Path(window.image_path).name == NAMES[0]
        assert window.annot_is_absent_background
        assert _on_screen(window, root) == (0, GREEN_STROKE)
        assert window.review_widget.decision() == 'absent'

        # Leaving, exporting or closing saves the scan again; none of that
        # may touch the file (the trainer would see changed annotations) or
        # the review.
        _next(window)
        _previous(window)
        assert window.save_annotation()
        assert path.stat().st_mtime_ns == written
        assert load_review(project, NAMES[0]) == record
        assert _counts(path) == (0, PIXELS)
        assert not dialogs
    finally:
        _close(window)


def check_changing_the_decision_puts_the_strokes_back(root, dialogs):
    _sync, project, window = _session(root)
    try:
        # Scan 1: green strokes, absent. Scan 2: no strokes, absent.
        _paint_green(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        for name in NAMES[:2]:
            assert _counts(_annotation(project, name)[1]) == (0, PIXELS)

        # Scan 2 becomes uncertain: with no strokes there is no file at all.
        _previous(window)
        assert Path(window.image_path).name == NAMES[1]
        assert _on_screen(window, root) == (0, 0)
        _decide(window, 'uncertain')
        _previous(window)
        assert _annotation(project, NAMES[1]) == (None, None)
        assert load_review(project, NAMES[1])['decision'] == 'uncertain'

        # Scan 1 becomes present: the lesion is painted and the file holds
        # the user's strokes again, on the same side.
        assert Path(window.image_path).name == NAMES[0]
        _paint_red(window)
        _decide(window, 'present')
        _next(window)
        side, path = _annotation(project, NAMES[0])
        assert side == 'train'
        assert _counts(path) == (RED_STROKE, GREEN_STROKE)
        assert load_review(project, NAMES[0])['decision'] == 'present'

        # And back to absent once the red mark is erased.
        _previous(window)
        assert not window.annot_is_absent_background
        painter = QtGui.QPainter(window.scene.annot_pixmap)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.fillRect(270, 210, 30, 12, QtGui.QColor(0, 0, 0, 0))
        painter.end()
        _decide(window, 'absent')
        _next(window)
        assert _counts(_annotation(project, NAMES[0])[1]) == (0, PIXELS)
        assert not dialogs
    finally:
        _close(window)


def check_red_marks_on_an_absent_scan_are_never_replaced(root, dialogs):
    _sync, project, window = _session(root)
    try:
        # Red and green strokes with an absent decision: blocked, as before.
        _paint(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        assert dialogs == ['Review not marked complete']
        assert Path(window.image_path).name == NAMES[0]
        assert not load_review(project, NAMES[0])['review_complete']
        assert _counts(_annotation(project, NAMES[0])[1]) == (
            RED_STROKE, GREEN_STROKE)
        del dialogs[:]

        # A scan already stored as clean, then given a red mark.
        window.review_widget.set_review('uncertain', True)
        _next(window)
        assert Path(window.image_path).name == NAMES[1]
        window.review_widget.set_review('absent', True)
        _next(window)
        assert _counts(_annotation(project, NAMES[1])[1]) == (0, PIXELS)
        _previous(window)
        _paint_red(window)
        _next(window)
        assert 'Review not marked complete' in dialogs
        assert Path(window.image_path).name == NAMES[1]
        assert not window.annot_is_absent_background
        assert _counts(_annotation(project, NAMES[1])[1]) == (RED_STROKE, 0)
    finally:
        _close(window)


def check_projects_without_the_option_keep_sparse_strokes(root, dialogs):
    _sync, project, window = _session(root, enabled=False)
    try:
        assert not window.absent_scans_as_background
        window.review_widget.set_review('absent', True)
        _next(window)
        assert _annotation(project, NAMES[0]) == (None, None)

        _paint_green(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        _side, path = _annotation(project, NAMES[1])
        assert _counts(path) == (0, GREEN_STROKE)
        _previous(window)
        assert not window.annot_is_absent_background
        assert _on_screen(window, root) == (0, GREEN_STROKE)
        assert not dialogs
    finally:
        _close(window)


def check_reopening_the_project_keeps_the_scan_clean(root, dialogs):
    sync, project, window = _session(root)
    try:
        _paint_green(window)
        window.review_widget.set_review('absent', True)
        _next(window)
        _paint(window)
        window.review_widget.set_review('present', True)
        _next(window)
    finally:
        _close(window)
    _side, path = _annotation(project, NAMES[0])
    written = path.stat().st_mtime_ns
    record = load_review(project, NAMES[0])

    window = _open(sync)
    try:
        # It resumes after the last scan with an annotation file.
        assert Path(window.image_path).name == NAMES[2]
        _previous(window)
        _previous(window)
        assert Path(window.image_path).name == NAMES[0]
        assert window.annot_is_absent_background
        assert _on_screen(window, root) == (0, GREEN_STROKE)
    finally:
        _close(window)
    assert path.stat().st_mtime_ns == written
    assert _counts(path) == (0, PIXELS)
    assert load_review(project, NAMES[0]) == record
    assert not dialogs


CHECKS = [
    check_absent_review_makes_the_whole_scan_background,
    check_returning_shows_own_strokes_and_rewrites_nothing,
    check_changing_the_decision_puts_the_strokes_back,
    check_red_marks_on_an_absent_scan_are_never_replaced,
    check_projects_without_the_option_keep_sparse_strokes,
    check_reopening_the_project_keeps_the_scan_clean,
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
