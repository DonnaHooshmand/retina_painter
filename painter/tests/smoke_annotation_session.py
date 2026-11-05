"""Opt-in synthetic GUI workflow check with an isolated local trainer sync root.

Run prepare, run the real trainer against the same sync root, then run verify.
This deliberately avoids main.py and never reads or writes user settings.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import site
import sys
import threading

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
GUI_SRC = REPO / "painter/src/main/python"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    import PyQt5  # noqa: F401
except ImportError:
    # The workstation has compatible Qt packages in the painter environment.
    # This is a process-local import path, not an environment repair.
    site.addsitedir(str(REPO / "painter/env/Lib/site-packages"))
sys.path.insert(0, str(GUI_SRC))

from PyQt5 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402
import retina_painter  # noqa: E402
from review_utils import (  # noqa: E402
    ANNOTATION_FOREGROUND_ONLY,
    completed_review_count,
    load_review,
    reconstruct_corrected_binary,
    training_has_started,
)


MARKER = "synthetic_gui_smoke.json"
NAMES = [f"synthetic_{index:02d}.png" for index in range(1, 13)]


def _paths(sync):
    project = sync / "projects/synthetic_session"
    return project, project / "synthetic_session.seg_proj", sync / "datasets/synthetic/images"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paint(window):
    painter = QtGui.QPainter(window.scene.annot_pixmap)
    painter.fillRect(270, 210, 30, 12, QtGui.QColor(255, 0, 0, 180))
    painter.fillRect(80, 400, 80, 12, QtGui.QColor(0, 255, 0, 180))
    painter.end()
    window.scene.annot_pixmap_holder.setPixmap(window.scene.annot_pixmap)


def _create(sync):
    project, config_path, images = _paths(sync)
    if project.exists() or images.exists() or (sync / MARKER).exists():
        raise ValueError("Prepare refuses an existing synthetic session")
    sync.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True)
    for directory in ("instructions", "executed_instructions", "failed_instructions"):
        (sync / directory).mkdir(exist_ok=True)
    for directory in ("annotations/train", "annotations/val", "models", "segmentations",
                      "reviews", "review_exports", "messages", "logs"):
        (project / directory).mkdir(parents=True)
    for index, name in enumerate(NAMES):
        array = np.full((600, 600, 3), 30, dtype=np.uint8)
        array[160:280] = 70
        array[190:250, 260:320] = 225
        array[300:310] = 100 + index
        Image.fromarray(array).save(images / name)
        if index < 10:
            # Deliberately unrelated pretraining prediction verifies exports
            # retain only clinician red marks during warm-up.
            segmentation = np.zeros((600, 600, 4), dtype=np.uint8)
            segmentation[10:20, 10:20] = [0, 255, 255, 180]
            Image.fromarray(segmentation).save(project / "segmentations" / name)
    val = [NAMES[4], NAMES[9]]
    config = {
        "name": "synthetic_session", "location": "projects/synthetic_session",
        "dataset": str(images), "file_names": NAMES,
        "train_file_names": [name for name in NAMES if name not in val],
        "val_file_names": val, "model_type": "unet", "training_seed": 0,
        "image_order_seed": 0, "auto_start_training_after_reviews": 10,
        "original_model_file": "random weights",
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    marker = {"synthetic_only": True, "version": 1,
              "image_hashes": {name: _sha(images / name) for name in NAMES}}
    (sync / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def _open(sync):
    window = retina_painter.RetinaPainter(sync)
    window.open_project(str(_paths(sync)[1]))
    return window


def prepare(sync, dialogs):
    _create(sync)
    project, _, _ = _paths(sync)
    window = _open(sync)
    sent = []
    original_send = window.send_instruction
    def record_send(name, content):
        sent.append((name, dict(content)))
        return original_send(name, content)
    window.send_instruction = record_send
    try:
        # Navigation is exercised through the real Qt button signal.
        window.nav.next_image_button.click()
        assert Path(window.image_path).name == NAMES[0]
        assert dialogs[-1][0] == "Review not complete"
        window.review_widget.set_review("present", True)
        window.nav.next_image_button.click()
        assert not load_review(project, NAMES[0])["review_complete"]
        assert dialogs[-1][0] == "Review not marked complete"
        for index in range(10):
            assert Path(window.image_path).name == NAMES[index]
            assert window.nav.next_image_button.isEnabled()
            _paint(window)
            window.review_widget.set_review("present", True)
            window.nav.next_image_button.click()
            assert completed_review_count(project) == index + 1
            assert Path(window.image_path).name == NAMES[index + 1]
            assert training_has_started(project) == (index == 9)
            assert window.training_has_started == (index == 9)
            record = load_review(project, NAMES[index])
            assert record["reconstruction_mode"] == ANNOTATION_FOREGROUND_ONLY
        starts = [content for name, content in sent if name == "start_training"]
        assert len(starts) == 1
        instruction = starts[0]
        assert instruction["model_type"] == "unet"
        assert instruction["training_seed"] == 0
        assert len(list((project / "annotations/train").glob("*.png"))) == 8
        assert len(list((project / "annotations/val").glob("*.png"))) == 2
        assert window.messages_label.text() == "Starting training..."
        assert not window.nav.next_image_button.isEnabled()
        window.segment_image([NAMES[10], NAMES[11]])
        return {"phase": "prepare", "completed_reviews": 10,
                "training_annotations": 8, "validation_annotations": 2,
                "autostart_instructions": 1, "pretraining_prediction_ignored": True,
                "incomplete_and_missing_red_reviews_blocked": True}
    finally:
        window.close()


def verify(sync, dialogs):
    marker_path = sync / MARKER
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["synthetic_only"] is True and marker["version"] == 1
    project, _, images = _paths(sync)
    assert completed_review_count(project) == 10
    for name, expected_hash in marker["image_hashes"].items():
        assert _sha(images / name) == expected_hash
    for name in NAMES[10:]:
        with Image.open(project / "segmentations" / name) as segmentation:
            assert segmentation.size == (600, 600)
    window = _open(sync)
    try:
        assert Path(window.image_path).name == NAMES[10]
        assert window.training_has_started
        assert not window.seg_pixmap.isNull()
        assert window.seg_pixmap.size() == QtCore.QSize(600, 600)
        assert window.nav.next_image_button.isEnabled()
        # Exercise the actual asynchronous prediction/status watcher. The
        # segmentation is the real trainer result, not fabricated here.
        window.seg_mtime = None
        window.nav.next_image_button.setEnabled(False)
        status = "Synthetic training status received"
        (project / "messages" / status).touch()
        QtTest.QTest.qWait(650)
        assert window.nav.next_image_button.isEnabled()
        assert window.messages_label.text() == status
        for index in (10, 11):
            assert Path(window.image_path).name == NAMES[index]
            _paint(window)
            window.review_widget.set_review("present", True)
            if index == 10:
                window.nav.next_image_button.click()
            else:
                assert window.save_annotation()
        assert completed_review_count(project) == 12
        window.close()
        window = _open(sync)
        assert Path(window.image_path).name == NAMES[0]
        assert window.review_widget.decision() == "present"
        assert window.review_widget.is_complete()
        # Opening an already-completed warm-up scan must not change its saved
        # pretraining provenance merely because training has since begun.
        before = load_review(project, NAMES[0])
        window.export_reviews()
        assert dialogs[-1][0] == "Completed reviews exported"
        exports = list((project / "review_exports").iterdir())
        assert len(exports) == 1
        with (exports[0] / "reviews.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 12
        # Compare the frozen exports with their immutable snapshots and
        # reconstruction mode; do not compare against a subsequently changed
        # model output from the live segmentation folder.
        for row in rows:
            record = load_review(project, row["filename"])
            mask_path = exports[0] / row["binary_mask"]
            with Image.open(mask_path) as mask:
                actual = np.array(mask) > 0
            if record["reconstruction_mode"] == ANNOTATION_FOREGROUND_ONLY:
                with Image.open(project / record["annotation_snapshot"]) as annotation:
                    pixels = np.array(annotation)
                expected = (pixels[:, :, 0] > 0) & (pixels[:, :, 3] > 0)
            else:
                expected = reconstruct_corrected_binary(
                    project / record["segmentation_snapshot"],
                    project / record["annotation_snapshot"])
            assert np.array_equal(actual, expected)
        after = load_review(project, NAMES[0])
        assert after == before, "Export silently revised an unchanged completed review"
        return {"phase": "verify", "completed_reviews": 12,
                "exported_reviews": len(rows), "export_snapshot_masks_match": True,
                "gui_prediction_loading": True, "gui_status_polling": True,
                "save_close_reopen": True,
                "unchanged_completed_review_preserved": True,
                "warmup_mode_before_reopen_export": before["reconstruction_mode"],
                "warmup_mode_after_reopen_export": after["reconstruction_mode"]}
    finally:
        window.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syncdir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "verify"), required=True)
    args = parser.parse_args()
    sync = args.syncdir.resolve()
    # Disable launch discovery so no bundled server or user sync is touched.
    retina_painter.find_trainer_launch = lambda: None
    sys.argv = [sys.argv[0]]
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialogs = []
    for name in ("warning", "about", "information"):
        setattr(QtWidgets.QMessageBox, name,
                lambda parent, title, message, *unused: dialogs.append((title, message)))
    original_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    try:
        result = prepare(sync, dialogs) if args.phase == "prepare" else verify(sync, dialogs)
        (sync / f"gui_{args.phase}_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        sys.excepthook = original_hook
        threading.excepthook = original_thread_hook
        app.quit()


if __name__ == "__main__":
    main()
