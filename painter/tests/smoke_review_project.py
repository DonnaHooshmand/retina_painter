"""Open an error-review project in the real painter window (synthetic data).

``trainer/experiments/prepare_error_review.py`` builds small projects for
looking at the scans several models get wrong. This check builds one from
synthetic images with that tool, opens it in ``RetinaPainter`` offscreen,
steps through every scan and confirms that

* a project whose dataset lives outside the ``datasets`` folder
  (``../curated_datasets/...``) opens at all;
* it opens on the first scan and shows the strokes and the overlay;
* moving on needs a decision, as in any review;
* training is never started, however many reviews are completed.

Run directly with the painter environment (PyQt5, no pytest needed):

    python smoke_review_project.py
"""

import contextlib
import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'painter' / 'src' / 'main' / 'python'))
sys.path.insert(0, str(REPO / 'trainer' / 'experiments'))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt5 import QtWidgets  # noqa: E402

import retina_painter  # noqa: E402
from review_utils import completed_review_count, training_has_started  # noqa: E402
import mask_io  # noqa: E402
import prepare_error_review  # noqa: E402

SHAPE = (120, 160)
NAMES = [f'DEMO{index:02d}_scan.png' for index in range(12)]


@contextlib.contextmanager
def patched_painter():
    """No trainer launch, no file watching, dialogs recorded not shown."""
    if QtWidgets.QApplication.instance() is None:
        patched_painter.app = QtWidgets.QApplication(sys.argv[:1])
    dialogs = []
    saved = [(retina_painter, 'find_trainer_launch'),
             (retina_painter.RetinaPainter, 'track_changes')]
    saved += [(QtWidgets.QMessageBox, kind)
              for kind in ('warning', 'about', 'information')]
    originals = [(owner, name, owner.__dict__[name]) for owner, name in saved]
    retina_painter.find_trainer_launch = lambda: None
    retina_painter.RetinaPainter.track_changes = lambda self: None
    for kind in ('warning', 'about', 'information'):
        setattr(QtWidgets.QMessageBox, kind, staticmethod(
            lambda parent, title, text, *rest: dialogs.append(title)))
    try:
        yield dialogs
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _synthetic_review_project(sync):
    """Six scans decided present and six absent, in a curated-dataset layout."""
    dataset = sync / 'curated_datasets' / 'demo'
    (sync / 'datasets').mkdir(parents=True)
    for folder in ('instructions', 'executed_instructions',
                   'failed_instructions'):
        (sync / folder).mkdir()
    (dataset / 'images').mkdir(parents=True)
    (dataset / 'ground_truth').mkdir()
    labels = []
    for index, name in enumerate(NAMES):
        Image.fromarray(np.full(SHAPE + (3,), 60 + index, np.uint8)).save(
            dataset / 'images' / name)
        present = index < 6
        labels.append(dict(filename=name, patient_id=f'D{index}', split='val',
                           decision='present' if present else 'absent'))
        if present or index % 2 == 0:
            foreground = np.zeros(SHAPE, bool)
            background = np.zeros(SHAPE, bool)
            if present:
                foreground[40:46, 50:60] = True
            background[100:104, 10:60] = True
            mask_io.write_annotation(
                dataset / 'annotations' / 'val' / name, foreground, background)
    with open(dataset / 'ground_truth' / 'bscan_labels.csv', 'w',
              newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    scans = [dict(filename=row['filename'],
                  clinician_decision=row['decision'],
                  group=('missed_positive' if row['decision'] == 'present'
                         else 'flagged_negative'),
                  models_flagging=0, models_compared=6) for row in labels]
    overlay = np.zeros(SHAPE, bool)
    overlay[20:30, 100:120] = True
    prepare_error_review.create_review_project(
        dataset, sync, 'review_demo', scans, lambda _name: overlay)
    return sync / 'projects' / 'review_demo', labels


def check_review_project_opens_steps_and_never_trains(root, dialogs):
    sync = root / 'sync'
    project, labels = _synthetic_review_project(sync)
    window = retina_painter.RetinaPainter(sync)
    sent = []
    send = window.send_instruction

    def record(name, content):
        sent.append(name)
        return send(name, content)

    window.send_instruction = record
    try:
        window.open_project(str(project / 'review_demo.seg_proj'))
        # The dataset is outside sync/datasets; its path must be normalised.
        assert '..' not in Path(window.image_path).parts
        assert Path(window.image_path).name == NAMES[0]
        assert os.path.isfile(window.seg_path)
        assert os.path.isfile(window.annot_path)

        window.nav.next_image_button.click()
        assert Path(window.image_path).name == NAMES[0]
        assert dialogs[-1] == 'Review not complete'

        for index, name in enumerate(NAMES):
            assert Path(window.image_path).name == name
            window.review_widget.set_review(labels[index]['decision'], True)
            window.nav.next_image_button.click()
            assert completed_review_count(project) == index + 1
        assert Path(window.image_path).name == NAMES[0]  # wrapped round

        # The painter asks for the current scan to be segmented on each
        # step; it must never ask for training.
        assert sent and set(sent) == {'segment'}
        assert not training_has_started(project)
        assert not window.training_has_started
        assert not list((project / 'annotations' / 'train').iterdir())
        assert not list((project / 'models').iterdir())
    finally:
        window.close()
        window.log_debounce.stop()


def main():
    with patched_painter() as dialogs, \
            tempfile.TemporaryDirectory() as directory:
        check_review_project_opens_steps_and_never_trains(
            Path(directory), dialogs)
    print('ok   check_review_project_opens_steps_and_never_trains')


if __name__ == '__main__':
    main()
