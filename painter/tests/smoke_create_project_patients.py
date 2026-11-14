"""Qt checks for the patient controls of the New Project dialog.

The last check covers the dialog's other project option, "Use scans marked
RIPL absent as clean scans", which is off unless ticked.

All data here is synthetic. Run directly with the painter environment, which
has PyQt5 but not necessarily pytest:

    python smoke_create_project_patients.py

The same checks are collected by pytest through
``test_create_project_patients.py`` wherever pytest and PyQt5 are both
installed.
"""

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[1] / 'src' / 'main' / 'python'
sys.path.insert(0, str(SRC))

from PyQt5 import QtWidgets  # noqa: E402

import create_project  # noqa: E402
from project_order import fixed_train_val_split, seeded_file_order  # noqa: E402


def _dataset(sync, names):
    directory = sync / 'datasets' / 'scans'
    directory.mkdir(parents=True)
    for name in names:
        # The dialog only looks at file names.
        (directory / name).write_bytes(b'')
    return directory


def _dialog(sync, directory, name='proj'):
    if QtWidgets.QApplication.instance() is None:
        _dialog.app = QtWidgets.QApplication(sys.argv[:1])
    widget = create_project.CreateProjectWidget(sync)
    widget.name_edit_widget.name_edit.setText(name)
    widget.set_image_directory(str(directory))
    widget.validate()
    return widget


def _project(sync, name='proj'):
    path = sync / 'projects' / name / f'{name}.seg_proj'
    return json.loads(path.read_text())


def _patient_names(patients=12, scans=3):
    return [f'P{patient:02d}_OD_{scan}.png'
            for patient in range(patients) for scan in range(scans)]


def _select(combo, data):
    combo.setCurrentIndex(combo.findData(data))


def check_patient_part_is_proposed_and_project_split_is_patient_pure(root):
    sync = root / 'sync'
    names = _patient_names()
    widget = _dialog(sync, _dataset(sync, names))

    assert widget.patient_source_combo.currentData() == widget.PATIENT_BY_NAME
    assert widget.patient_part_combo.currentData() == 0
    assert widget.patient_part_combo.itemText(0) == 'Part 1: P00'
    preview = widget.patient_preview_label.text()
    assert '36 images from 12 patients' in preview
    assert 'median 3 per patient, largest 3' in preview
    assert widget.patient_problem is None
    assert widget.create_project_btn.isEnabled()

    # The seed changes who is where, never whether patients stay together.
    widget.image_order_seed_spin.setValue(5)
    assert widget.patient_problem is None
    widget.create_project()

    project = _project(sync)
    assert project['image_order_seed'] == 5
    assert project['split_method'] == (
        'patient_grouped_seeded_order_5_train_to_1_val_v1')
    assert project['patient_rule'] == {
        'source': 'filename_part', 'part': 0, 'separators': '_- .'}
    train, val = project['train_file_names'], project['val_file_names']
    assert sorted(train + val) == sorted(names)
    train_patients = {name.split('_')[0] for name in train}
    val_patients = {name.split('_')[0] for name in val}
    assert train_patients and val_patients
    assert train_patients.isdisjoint(val_patients)
    # One opaque number per image: same number, same patient; no identifiers.
    groups = dict(zip(project['file_names'], project['patient_groups']))
    assert all(isinstance(number, int) for number in groups.values())
    assert len(set(groups.values())) == 12
    for name in names:
        assert groups[name] == groups[name.split('_')[0] + '_OD_0.png']


def check_a_part_that_cannot_separate_patients_blocks_creation(root):
    sync = root / 'sync'
    widget = _dialog(sync, _dataset(sync, _patient_names() + ['odd.png']))

    _select(widget.patient_part_combo, 1)  # "OD" in every name but one
    assert '1 of 37 file names have no part 2' in widget.patient_problem
    assert not widget.create_project_btn.isEnabled()
    assert widget.info_label.text() == widget.patient_problem

    _select(widget.patient_part_combo, 0)
    assert widget.patient_problem is None
    assert widget.create_project_btn.isEnabled()
    assert not (sync / 'projects').exists()


def check_one_patient_for_everything_blocks_creation(root):
    sync = root / 'sync'
    widget = _dialog(sync, _dataset(sync, _patient_names()))

    _select(widget.patient_part_combo, 1)  # "OD" for every image
    assert 'one patient' in widget.patient_preview_label.text()
    assert widget.patient_problem
    assert not widget.create_project_btn.isEnabled()


def check_names_without_a_patient_part_keep_the_per_image_split(root):
    sync = root / 'sync'
    names = [f'img{index:03d}.png' for index in range(24)]
    widget = _dialog(sync, _dataset(sync, names))

    assert widget.patient_source_combo.currentData() == widget.PATIENT_NONE
    assert widget.patient_name_widget.isHidden()
    assert 'one by one' in widget.patient_preview_label.text()
    widget.create_project()

    project = _project(sync)
    ordered = seeded_file_order(names, 0)
    assert project['file_names'] == ordered
    assert [project['train_file_names'], project['val_file_names']] == list(
        fixed_train_val_split(ordered))
    assert project['split_method'] == 'seeded_order_5_train_to_1_val_v1'
    assert project['patient_rule'] == {'source': 'none'}
    assert project['patient_groups'] is None


def check_patient_list_csv_drives_the_split(root):
    sync = root / 'sync'
    names = [f'img{index:03d}.png' for index in range(24)]
    widget = _dialog(sync, _dataset(sync, names))
    patient = {name: f'patient {index // 4}'
               for index, name in enumerate(names)}

    def write(path, rows):
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(
                stream, fieldnames=['filename', 'patient_id'])
            writer.writeheader()
            writer.writerows(
                {'filename': name, 'patient_id': patient[name]}
                for name in rows)
        return path

    _select(widget.patient_source_combo, widget.PATIENT_BY_CSV)
    assert widget.patient_csv_widget.isHidden() is False
    assert 'Choose a patient list' in widget.patient_problem
    assert not widget.create_project_btn.isEnabled()

    widget.set_patient_csv(str(write(root / 'partial.csv', names[1:])))
    assert '1 of 24 images have no patient' in widget.patient_problem
    assert not widget.create_project_btn.isEnabled()

    widget.set_patient_csv(str(write(root / 'patients.csv', names)))
    assert widget.patient_problem is None
    assert '24 images from 6 patients' in widget.patient_preview_label.text()
    widget.create_project()

    project = _project(sync)
    assert project['patient_rule'] == {'source': 'csv', 'file': 'patients.csv'}
    train = {patient[name] for name in project['train_file_names']}
    val = {patient[name] for name in project['val_file_names']}
    assert train and val and train.isdisjoint(val)


def check_separator_choice_changes_the_parts(root):
    sync = root / 'sync'
    names = [f'AB-{patient:02d}_OD_{scan}.png'
             for patient in range(6) for scan in range(3)]
    widget = _dialog(sync, _dataset(sync, names))

    # With the hyphen as a separator the identifier is cut in two.
    assert widget.patient_part_combo.itemText(0) == 'Part 1: AB'
    assert widget.patient_part_combo.currentData() == 1

    dict(widget.separator_checks)['-'].setChecked(False)
    assert widget.patient_part_combo.itemText(0) == 'Part 1: AB-00'
    assert widget.patient_part_combo.currentData() == 0
    assert '18 images from 6 patients' in widget.patient_preview_label.text()
    assert widget.current_patient_grouping(sorted(names))[1] == {
        'source': 'filename_part', 'part': 0, 'separators': '_ .'}


def check_absent_scans_option_is_off_unless_ticked(root):
    sync = root / 'sync'
    directory = _dataset(sync, _patient_names())

    widget = _dialog(sync, directory, name='default')
    assert not widget.absent_background_checkbox.isChecked()
    widget.create_project()
    assert _project(sync, 'default')['absent_scans_as_background'] is False

    widget = _dialog(sync, directory, name='clean_scans')
    widget.absent_background_checkbox.setChecked(True)
    widget.create_project()
    assert _project(sync, 'clean_scans')['absent_scans_as_background'] is True


CHECKS = [
    check_patient_part_is_proposed_and_project_split_is_patient_pure,
    check_a_part_that_cannot_separate_patients_blocks_creation,
    check_one_patient_for_everything_blocks_creation,
    check_names_without_a_patient_part_keep_the_per_image_split,
    check_patient_list_csv_drives_the_split,
    check_separator_choice_changes_the_parts,
    check_absent_scans_option_is_off_unless_ticked,
]


def main():
    def unexpected_dialog(_parent, title, text, *_rest):
        raise AssertionError(f'Unexpected dialog: {title}: {text}')

    QtWidgets.QMessageBox.about = unexpected_dialog
    for check in CHECKS:
        with tempfile.TemporaryDirectory() as directory:
            check(Path(directory))
        print('ok  ', check.__name__)
    print(f'{len(CHECKS)} checks passed')


if __name__ == '__main__':
    main()
