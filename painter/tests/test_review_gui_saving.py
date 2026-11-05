"""Synthetic Qt regressions for automatic saves of completed reviews."""

import csv
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest


os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PyQt5")
sys.path.insert(0, str(Path(__file__).parent))
from smoke_annotation_session import (  # noqa: E402
    NAMES, QtGui, QtWidgets, _create, _open, _paint, _paths, retina_painter,
)
from review_utils import (  # noqa: E402
    ANNOTATION_FOREGROUND_ONLY, PREDICTION_CORRECTED, load_review,
    mark_training_started,
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    sync = tmp_path / "synthetic_sync"
    _create(sync)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(sys, "argv", ["synthetic_gui_test"])
    monkeypatch.setattr(retina_painter, "find_trainer_launch", lambda: None)
    monkeypatch.setattr(retina_painter.RetinaPainter, "track_changes", lambda self: None)
    dialogs = []
    for name in ("warning", "about", "information"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name,
                            lambda parent, title, text, *rest: dialogs.append(title))
    original_hook, original_thread_hook = sys.excepthook, threading.excepthook
    windows = []
    def open_window():
        window = _open(sync)
        windows.append(window)
        return window
    window = open_window()
    _paint(window)
    window.review_widget.set_review("present", True)
    assert window.save_annotation()
    project = _paths(sync)[0]
    before = load_review(project, NAMES[0])
    assert before["reconstruction_mode"] == ANNOTATION_FOREGROUND_ONLY
    try:
        yield window, project, before, dialogs, open_window
    finally:
        for current in windows:
            current.close()
            current.log_debounce.stop()
        sys.excepthook, threading.excepthook = original_hook, original_thread_hook
        app.quit()


def _start_and_change_prediction(window, project):
    mark_training_started(project)
    window.training_has_started = True
    pixels = np.zeros((600, 600, 4), dtype=np.uint8)
    pixels[100:120, 100:120] = [0, 255, 255, 180]
    Image.fromarray(pixels).save(project / "segmentations" / NAMES[0])
    window.update_seg()


def test_automatic_save_export_close_and_reopen_preserve_completed_snapshot(session):
    window, project, before, dialogs, open_window = session
    _start_and_change_prediction(window, project)
    assert window.save_annotation()
    assert load_review(project, NAMES[0]) == before
    window.export_reviews()
    assert dialogs == ["Completed reviews exported"]
    assert load_review(project, NAMES[0]) == before
    exports = list((project / "review_exports").iterdir())
    with (exports[0] / "reviews.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    with Image.open(exports[0] / row["binary_mask"]) as image:
        mask = np.array(image) > 0
    assert int(mask.sum()) == 360
    assert not mask[100:120, 100:120].any()
    window.close()
    assert load_review(project, NAMES[0]) == before
    reopened = open_window()
    reopened.nav.prev_image_button.click()
    assert Path(reopened.image_path).name == NAMES[0]
    assert reopened.save_annotation()
    assert load_review(project, NAMES[0]) == before


@pytest.mark.parametrize("edit", ["annotation", "recomplete", "decision"])
def test_intentional_review_revision_refreshes_prediction_snapshot(session, edit):
    window, project, before, _, _ = session
    _start_and_change_prediction(window, project)
    if edit == "annotation":
        painter = QtGui.QPainter(window.scene.annot_pixmap)
        painter.fillRect(330, 290, 10, 10, QtGui.QColor(255, 0, 0, 180))
        painter.end()
    elif edit == "recomplete":
        window.review_widget.complete_checkbox.setChecked(False)
        window.review_widget.complete_checkbox.setChecked(True)
    else:
        index = window.review_widget.decision_combo.findData("uncertain")
        window.review_widget.decision_combo.setCurrentIndex(index)
    assert window.save_annotation()
    after = load_review(project, NAMES[0])
    assert after["reconstruction_mode"] == PREDICTION_CORRECTED
    assert after["segmentation_sha256"] != before["segmentation_sha256"]
    if edit == "annotation":
        assert after["annotation_sha256"] != before["annotation_sha256"]
    if edit == "decision":
        assert after["decision"] == "uncertain"
    # A second automatic save is idempotent after the deliberate update too.
    assert window.save_annotation()
    assert load_review(project, NAMES[0]) == after


def test_newest_status_wins_even_when_old_message_sorts_last_and_stays_locked(
        session, monkeypatch):
    window, project, _, _, _ = session
    messages = project / "messages"
    old = messages / "Training started"
    new = messages / "Stopped"
    old.touch()
    new.touch()
    now = time.time_ns()
    os.utime(old, ns=(now - 2_000_000_000, now - 2_000_000_000))
    os.utime(new, ns=(now, now))
    original_remove = os.remove
    original_ls = retina_painter.ls
    def locked_remove(path):
        if Path(path) == old:
            raise PermissionError("Synthetic message is temporarily locked")
        return original_remove(path)
    with monkeypatch.context() as patch:
        patch.setattr(retina_painter.os, "remove", locked_remove)
        patch.setattr(retina_painter, "ls",
                      lambda path: ["Vanished message"] + sorted(original_ls(path)))
        window.update_training_messages()
        assert window.messages_label.text() == "Stopped"
        assert old.exists() and not new.exists()
        # A failed delete must not make an older status win the next poll.
        window.update_training_messages()
        assert window.messages_label.text() == "Stopped"
    window.update_training_messages()
    assert not old.exists()
    assert window.messages_label.text() == "Stopped"


def test_temporarily_missing_status_directory_does_not_stop_polling(session):
    window, project, _, _, _ = session
    window.messages_label.setText("Latest status")
    window.message_dir = project / "temporarily_unavailable_messages"
    window.update_training_messages()
    assert window.messages_label.text() == "Latest status"
