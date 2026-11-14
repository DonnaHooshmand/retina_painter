"""
Copyright (C) 2020 Abraham George Smith
Copyright (C) 2021 Abraham George Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

#pylint: disable=I1101,C0111,W0201,R0903,E0611, R0902, R0914
import csv
import os
import time
import shutil
import json

from pathlib import Path, PurePosixPath
from PyQt5 import QtWidgets
from PyQt5 import QtCore

from im_utils import is_image
import file_utils
from name_edit_widget import NameEditWidget
from palette import PaletteEditWidget
from project_order import (filename_parts, fixed_train_val_split,
                           grouping_summary, manifest_file_order,
                           patient_group_numbers,
                           patient_grouped_train_val_split,
                           patient_ids_from_csv,
                           patient_ids_from_filename_part, seeded_file_order,
                           suggest_patient_part)

class CreateProjectWidget(QtWidgets.QWidget):

    created = QtCore.pyqtSignal(Path)

    # Where the patient of each image comes from.
    PATIENT_BY_NAME = 'filename_part'
    PATIENT_BY_CSV = 'csv'
    PATIENT_NONE = 'none'

    def __init__(self, sync_dir):
        super().__init__()
        self.selected_dir = None
        self.proj_name = None
        self.selected_model = None
        self.use_random_weights = True
        self.model_type = 'unet'
        self.image_order_seed = 0
        self.sync_dir = sync_dir
        self.dataset_fnames = []
        self.patient_csv_path = None
        # A message here means the patient settings cannot produce a split.
        self.patient_problem = None
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.name_edit_widget = NameEditWidget('Project')
        self.name_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.name_edit_widget)

        self.add_im_dir_widget()
        self.add_image_order_seed_widget()
        self.add_patient_widget()
        self.add_absent_scans_widget()
        self.add_model_type_widget()
        self.add_radio_widget()
        self.add_model_btn()
        if False:
            self.add_palette_widget()
        self.add_info_label()
        self.add_create_btn()

    def add_im_dir_widget(self):
        # Add specify image directory button
        directory_label = QtWidgets.QLabel()
        directory_label.setText("Image directory: Not yet specified")
        self.layout.addWidget(directory_label)
        self.directory_label = directory_label

        specify_image_dir_btn = QtWidgets.QPushButton('Specify image directory')
        specify_image_dir_btn.clicked.connect(self.select_photo_dir)
        self.layout.addWidget(specify_image_dir_btn)

    def add_image_order_seed_widget(self):
        seed_widget = QtWidgets.QWidget()
        seed_layout = QtWidgets.QHBoxLayout()
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_widget.setLayout(seed_layout)

        seed_label = QtWidgets.QLabel("Trial seed:")
        seed_layout.addWidget(seed_label)

        self.image_order_seed_spin = QtWidgets.QSpinBox()
        self.image_order_seed_spin.setRange(0, 2_147_483_647)
        self.image_order_seed_spin.setValue(self.image_order_seed)
        self.image_order_seed_spin.setToolTip(
            "For new projects, the seed fixes image order, train/validation "
            "assignment, model initialization, and training sampling."
        )
        self.image_order_seed_spin.valueChanged.connect(
            self.on_image_order_seed_changed)
        seed_layout.addWidget(self.image_order_seed_spin)
        self.layout.addWidget(seed_widget)

        seed_hint = QtWidgets.QLabel(
            "Use the same dataset and seed for matched U-Net, RETFound, and "
            "RFA trials."
        )
        seed_hint.setWordWrap(True)
        self.layout.addWidget(seed_hint)

    def on_image_order_seed_changed(self, seed):
        self.image_order_seed = int(seed)
        if hasattr(self, 'patient_preview_label'):
            self.update_patient_preview()

    def add_patient_widget(self):
        """Controls that keep each patient on one side of the split."""
        box = QtWidgets.QGroupBox("Patient separation")
        box_layout = QtWidgets.QVBoxLayout()
        box.setLayout(box_layout)
        self.layout.addWidget(box)

        hint = QtWidgets.QLabel(
            "Keep every image of a patient together in training or in "
            "validation, so validation measures patients the model has not "
            "seen."
        )
        hint.setWordWrap(True)
        box_layout.addWidget(hint)

        self.patient_source_combo = QtWidgets.QComboBox()
        self.patient_source_combo.addItem(
            "Patient is part of the file name", self.PATIENT_BY_NAME)
        self.patient_source_combo.addItem(
            "Patient list (CSV with filename and patient_id columns)",
            self.PATIENT_BY_CSV)
        self.patient_source_combo.addItem(
            "No patient grouping (every image independent)",
            self.PATIENT_NONE)
        self.patient_source_combo.currentIndexChanged.connect(
            self.on_patient_source_changed)
        box_layout.addWidget(self.patient_source_combo)

        # Patient read from one part of the file name.
        self.patient_name_widget = QtWidgets.QWidget()
        name_layout = QtWidgets.QVBoxLayout()
        name_layout.setContentsMargins(0, 0, 0, 0)
        self.patient_name_widget.setLayout(name_layout)
        self.patient_example_label = QtWidgets.QLabel()
        self.patient_example_label.setWordWrap(True)
        name_layout.addWidget(self.patient_example_label)

        part_row = QtWidgets.QHBoxLayout()
        part_row.addWidget(QtWidgets.QLabel("Patient is:"))
        self.patient_part_combo = QtWidgets.QComboBox()
        self.patient_part_combo.setToolTip(
            "The part of the file name that is the same for every image of "
            "one patient.")
        self.patient_part_combo.currentIndexChanged.connect(
            self.update_patient_preview)
        part_row.addWidget(self.patient_part_combo, 1)
        name_layout.addLayout(part_row)

        separator_row = QtWidgets.QHBoxLayout()
        separator_row.addWidget(QtWidgets.QLabel("Parts are separated by:"))
        self.separator_checks = []
        for character, label in (('_', 'underscore _'), ('-', 'hyphen -'),
                                 (' ', 'space'), ('.', 'dot .')):
            check = QtWidgets.QCheckBox(label)
            check.setChecked(True)
            check.stateChanged.connect(self.on_separators_changed)
            separator_row.addWidget(check)
            self.separator_checks.append((character, check))
        separator_row.addStretch(1)
        name_layout.addLayout(separator_row)
        box_layout.addWidget(self.patient_name_widget)

        # Patient read from a list.
        self.patient_csv_widget = QtWidgets.QWidget()
        csv_layout = QtWidgets.QHBoxLayout()
        csv_layout.setContentsMargins(0, 0, 0, 0)
        self.patient_csv_widget.setLayout(csv_layout)
        csv_button = QtWidgets.QPushButton('Choose patient list')
        csv_button.clicked.connect(self.select_patient_csv)
        csv_layout.addWidget(csv_button)
        self.patient_csv_label = QtWidgets.QLabel("No patient list chosen")
        csv_layout.addWidget(self.patient_csv_label, 1)
        box_layout.addWidget(self.patient_csv_widget)

        self.patient_preview_label = QtWidgets.QLabel()
        self.patient_preview_label.setWordWrap(True)
        box_layout.addWidget(self.patient_preview_label)

        self.on_patient_source_changed()

    def current_separators(self):
        return ''.join(character for character, check
                       in self.separator_checks if check.isChecked())

    def on_patient_source_changed(self, _index=None):
        source = self.patient_source_combo.currentData()
        self.patient_name_widget.setVisible(source == self.PATIENT_BY_NAME)
        self.patient_csv_widget.setVisible(source == self.PATIENT_BY_CSV)
        self.update_patient_preview()

    def on_separators_changed(self, _state=None):
        self.refresh_patient_options(suggest=True)

    def list_dataset_images(self):
        if not self.selected_dir or not os.path.isdir(self.selected_dir):
            return []
        return sorted(name for name in file_utils.ls(self.selected_dir)
                      if is_image(name))

    def set_image_directory(self, directory):
        """Use *directory* as the dataset and propose a patient grouping."""
        self.selected_dir = directory
        self.directory_label.setText('Image directory: ' + self.selected_dir)
        self.dataset_fnames = self.list_dataset_images()
        by_name = suggest_patient_part(
            self.dataset_fnames, self.current_separators()) is not None
        # File names that share no part say nothing about patients, so start
        # from the per-image split instead of a grouping that does nothing.
        source = (self.PATIENT_BY_NAME if by_name or not self.dataset_fnames
                  else self.PATIENT_NONE)
        self.patient_source_combo.blockSignals(True)
        self.patient_source_combo.setCurrentIndex(
            self.patient_source_combo.findData(source))
        self.patient_source_combo.blockSignals(False)
        self.patient_name_widget.setVisible(source == self.PATIENT_BY_NAME)
        self.patient_csv_widget.setVisible(False)
        self.refresh_patient_options(suggest=True)

    def refresh_patient_options(self, suggest=False):
        """Rebuild the file-name part choices from an example file name."""
        separators = self.current_separators()
        previous = self.patient_part_combo.currentData()
        self.patient_part_combo.blockSignals(True)
        self.patient_part_combo.clear()
        if self.dataset_fnames:
            example = self.dataset_fnames[0]
            self.patient_example_label.setText(
                f"Example file name: {example}")
            for index, part in enumerate(filename_parts(example, separators)):
                self.patient_part_combo.addItem(
                    f"Part {index + 1}: {part}", index)
            chosen = None
            if suggest:
                chosen = suggest_patient_part(self.dataset_fnames, separators)
            if chosen is None:
                chosen = previous
            position = self.patient_part_combo.findData(chosen)
            self.patient_part_combo.setCurrentIndex(max(position, 0))
        else:
            self.patient_example_label.setText(
                "Choose an image directory to see an example file name.")
        self.patient_part_combo.blockSignals(False)
        self.update_patient_preview()

    def select_patient_csv(self):
        start_dir = self.selected_dir or str(self.sync_dir)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose patient list", start_dir, "CSV files (*.csv)")
        if file_path:
            self.set_patient_csv(file_path)

    def set_patient_csv(self, file_path):
        self.patient_csv_path = os.path.abspath(file_path)
        self.patient_csv_label.setText(os.path.basename(self.patient_csv_path))
        self.update_patient_preview()

    def ordered_file_names(self, dataset_path, fnames):
        """Navigation order for *fnames*, as stored in the project file.

        Returns ``(ordered names, order method, split-method stem, manifest
        path)``. Raises ``ValueError`` for an unusable dataset manifest.
        """
        manifest_order, manifest_path = manifest_file_order(
            dataset_path, fnames)
        if manifest_order is None:
            return (seeded_file_order(fnames, self.image_order_seed),
                    'seeded_shuffle_v1', 'seeded_order', None)
        # The manifest rank is already a reproducible random order. Using it
        # directly guarantees that the 150-image project is the exact prefix
        # of the 200-, 250-, ..., 600-image projects.
        return manifest_order, 'manifest_rank_v1', 'manifest_rank', manifest_path

    def current_patient_grouping(self, fnames):
        """Patients for *fnames* under the chosen settings.

        Returns ``(patient_of, rule, problem)``. ``patient_of`` is ``None``
        when images are split individually. ``problem`` is a message when the
        settings cannot give every image a patient; no project is created
        while it is set.
        """
        source = self.patient_source_combo.currentData()
        if source == self.PATIENT_NONE:
            return None, {'source': self.PATIENT_NONE}, None
        if source == self.PATIENT_BY_CSV:
            if not self.patient_csv_path:
                return None, None, "Choose a patient list to split by patient."
            try:
                patient_of = patient_ids_from_csv(self.patient_csv_path, fnames)
            except (OSError, ValueError, csv.Error) as error:
                return None, None, str(error)
            rule = {'source': self.PATIENT_BY_CSV,
                    'file': os.path.basename(self.patient_csv_path)}
            return patient_of, rule, None
        part = self.patient_part_combo.currentData()
        if part is None:
            return None, None, ("Choose which part of the file name is the "
                                "patient.")
        separators = self.current_separators()
        patient_of, unmatched = patient_ids_from_filename_part(
            fnames, part, separators)
        if unmatched:
            return None, None, (
                f"{len(unmatched)} of {len(fnames)} file names have no part "
                f"{part + 1}. Choose another part or separator.")
        rule = {'source': self.PATIENT_BY_NAME, 'part': part,
                'separators': separators}
        return patient_of, rule, None

    def update_patient_preview(self, _index=None):
        """Show what the chosen patient settings do, before creating."""
        fnames = self.dataset_fnames
        problem = None
        if not fnames:
            text = "Choose an image directory to preview the split."
        else:
            patient_of, _rule, problem = self.current_patient_grouping(fnames)
            if problem:
                text = problem
            elif patient_of is None:
                text = ("Images are split one by one: every sixth image is "
                        "validation, so one patient can be on both sides.")
            else:
                text, problem = self.describe_patient_split(fnames, patient_of)
        self.patient_problem = problem
        self.patient_preview_label.setText(text)
        self.patient_preview_label.setStyleSheet(
            "color: #b00020;" if problem else "")
        if hasattr(self, 'create_project_btn'):
            self.validate()

    def describe_patient_split(self, fnames, patient_of):
        """Return ``(preview text, problem)`` for a patient grouping."""
        summary = grouping_summary(fnames, patient_of)
        if summary['patients'] < 2:
            return ("All images belong to one patient with this choice, so "
                    "they cannot be split. Choose another part."), (
                        "The patient settings put every image in one patient.")
        try:
            ordered = self.ordered_file_names(self.selected_dir, fnames)[0]
        except ValueError as error:
            return str(error), str(error)
        train, val = patient_grouped_train_val_split(ordered, patient_of)
        lines = [
            f"{summary['images']} images from {summary['patients']} patients "
            f"(median {summary['median_images_per_patient']:g} per patient, "
            f"largest {summary['largest_patient_images']}).",
            f"Training: {len(train)} images from "
            f"{len({patient_of[name] for name in train})} patients. "
            f"Validation: {len(val)} images from "
            f"{len({patient_of[name] for name in val})} patients.",
        ]
        if summary['patients'] == summary['images']:
            lines.append(
                "Every image is its own patient with this choice, which is "
                "the same as no patient grouping. Check the selected part.")
        return "\n".join(lines), None

    def add_absent_scans_widget(self):
        """Opt-in: train on a scan reviewed as absent as background everywhere."""
        self.absent_background_checkbox = QtWidgets.QCheckBox(
            "Use scans marked RIPL absent as clean scans")
        self.absent_background_checkbox.setChecked(False)
        self.absent_background_checkbox.setToolTip(
            "When a review is completed as RIPL absent, every pixel of that "
            "scan is used as background for training and validation, without "
            "any painting. Off: only painted pixels are used, as in "
            "RootPainter."
        )
        self.layout.addWidget(self.absent_background_checkbox)

    def add_model_type_widget(self):
        label = QtWidgets.QLabel("Model type:")
        self.layout.addWidget(label)
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItem("U-Net (original RootPainter)", "unet")
        self.model_type_combo.addItem("RETFound + plain decoder", "retfound")
        self.model_type_combo.addItem("RETFound + RFA-U-Net (recommended)", "retfound_rfa")
        self.model_type_combo.addItem("FunduSegmenter (local adapter)", "fundusegmenter")
        self.model_type_combo.currentIndexChanged.connect(self.on_model_type_changed)
        self.layout.addWidget(self.model_type_combo)

        hint = QtWidgets.QLabel()
        hint.setWordWrap(True)
        hint.setOpenExternalLinks(True)
        hint.setVisible(False)
        self.layout.addWidget(hint)
        self.retfound_hf_hint_label = hint
        self.on_model_type_changed(self.model_type_combo.currentIndex())

    def on_model_type_changed(self, index):
        self.model_type = self.model_type_combo.itemData(index)
        if hasattr(self, "retfound_hf_hint_label"):
            if self.model_type in ("retfound", "retfound_rfa"):
                import retina_painter as rp

                page = rp.RETFOUND_WEIGHTS_HF_PAGE
                self.retfound_hf_hint_label.setText(
                    "RETFound setup: accept gated access on the Hub, then run "
                    f"<code>python setup_retfound.py --token …</code> from the repo root, "
                    f"or place <code>RETFound_oct.pth</code> in <code>~/.cache/retina_painter/</code>. "
                    f"Weights page: <a href=\"{page}\">{page}</a>"
                )
                self.retfound_hf_hint_label.setVisible(True)
            elif self.model_type == "fundusegmenter":
                self.retfound_hf_hint_label.setText(
                    "FunduSegmenter uses the local implementation in "
                    "<code>trainer/src/fundusegmenter_model.py</code>. "
                    "No external weight download is required for random-weight startup."
                )
                self.retfound_hf_hint_label.setVisible(True)
            else:
                self.retfound_hf_hint_label.setVisible(False)

    def add_radio_widget(self):
        radio_widget = QtWidgets.QWidget()
        radio_layout = QtWidgets.QHBoxLayout()
        radio_widget.setLayout(radio_layout)
        self.layout.addWidget(radio_widget)

        # Add radio, use random weight or specify model file.
        radio = QtWidgets.QRadioButton("Random Weights")
        radio.setChecked(True)
        radio.name = "random"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

        radio = QtWidgets.QRadioButton("Specify Model")
        radio.name = "specify"
        radio.toggled.connect(self.on_radio_clicked)
        radio_layout.addWidget(radio)

    def add_model_btn(self):
        model_label = QtWidgets.QLabel()
        model_label.setText("Model: Please specify model file")
        self.layout.addWidget(model_label)
        self.model_label = model_label
        specify_model_btn = QtWidgets.QPushButton('Specify model file')
        specify_model_btn.clicked.connect(self.select_model)
        self.specify_model_btn = specify_model_btn
        self.layout.addWidget(specify_model_btn)

        self.model_label.setVisible(False)
        self.specify_model_btn.setVisible(False)

    def add_palette_widget(self):
        self.palette_edit_widget = PaletteEditWidget()
        self.palette_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.palette_edit_widget)

    def add_info_label(self):
        info_label = QtWidgets.QLabel()
        info_label.setText("Name, directory and model must be specified"
                           " to create project.")
        self.layout.addWidget(info_label)
        self.info_label = info_label

    def add_create_btn(self):
        # Add create button
        create_project_btn = QtWidgets.QPushButton('Create project')
        create_project_btn.clicked.connect(self.create_project)
        self.layout.addWidget(create_project_btn)
        create_project_btn.setEnabled(False)
        self.create_project_btn = create_project_btn

    def on_radio_clicked(self):
        radio = self.sender()
        if radio.isChecked():
            print("Radio is %s" % (radio.name))
            specify = (radio.name == 'specify')
            self.model_label.setVisible(specify)
            self.specify_model_btn.setVisible(specify)
            self.use_random_weights = not specify
            self.validate()

    def validate(self):
        self.proj_name = self.name_edit_widget.name
        if not self.proj_name:
            self.info_label.setText("Name must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        if not self.selected_dir:
            self.info_label.setText("Directory must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        if not self.use_random_weights and not self.selected_model:
            self.info_label.setText("Starting model must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return

        cur_files = os.listdir(self.selected_dir)
        cur_files = [is_image(f) for f in cur_files]
        if not cur_files:
            message = "Folder contains no images."
            self.info_label.setText(message)
            self.create_project_btn.setEnabled(False)
            return

        if self.patient_problem:
            self.info_label.setText(self.patient_problem)
            self.create_project_btn.setEnabled(False)
            return

        if False:
            if len(self.palette_edit_widget.get_brush_data()) < 2:
                self.info_label.setText('At least one foreground class must be specified')
                self.create_project_btn.setEnabled(False)
                return

        self.project_location = os.path.join('projects', self.proj_name)
        if os.path.exists(os.path.join(self.sync_dir, self.project_location)):
            self.info_label.setText(f"Project with name {self.proj_name} already exists")
            self.create_project_btn.setEnabled(False)
        else:
            self.info_label.setText(f"Project location: {self.project_location}")
            self.create_project_btn.setEnabled(True)


    def select_photo_dir(self):
        self.photo_dialog = QtWidgets.QFileDialog(self, directory=os.path.join(self.sync_dir, 'datasets'))

        self.photo_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def output_selected():
            self.set_image_directory(self.photo_dialog.selectedFiles()[0])
            self.validate()

        self.photo_dialog.fileSelected.connect(output_selected)
        self.photo_dialog.open()


    def select_model(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self,
                                                             "Specify model file", "",
                                                             "Pickle Files (*.pkl)",
                                                             options=options)
        if file_path:
            file_path = os.path.abspath(file_path)
            self.selected_model = file_path
            self.model_label.setText('Model file: ' + self.selected_model)
            self.validate()

    def create_project(self):
        project_name = self.proj_name
        project_location = Path(self.project_location)

        dataset_path = os.path.abspath(self.selected_dir)
        datasets_dir = str(self.sync_dir / 'datasets')
    
        if not dataset_path.startswith(datasets_dir):
            message = ("When creating a project the selected dataset must be in "
                       "the datasets folder. The selected dataset is "
                       f"{dataset_path} and the datasets folder is "
                       f"{datasets_dir}.\n\n"
                       "Your sync directory is currently specified as "
                       f"{self.sync_dir}. Your active datasets and projects must"
                       " be located in this folder."
                       " If you would like to modify your local sync directory"
                       " then this can be done using the 'Specify sync directory'"
                       " option availble from the extras menu in the RetinaPainter GUI.")
        
            QtWidgets.QMessageBox.about(self, 'Project Creation Error', message)
            return

        # Store a reproducible navigation order and pre-assign every filename
        # to train or validation. The assignment must not depend on whether a
        # particular model happened to need a correction on an earlier scan.
        # This is worked out before any folder is created, so a problem with
        # the manifest or the patient settings leaves nothing half-made.
        all_fnames = [a for a in file_utils.ls(dataset_path) if is_image(a)]
        try:
            (all_fnames, image_order_method, order_name,
             manifest_path) = self.ordered_file_names(dataset_path, all_fnames)
        except ValueError as error:
            QtWidgets.QMessageBox.about(
                self, 'Project Creation Error', str(error))
            return
        patient_of, patient_rule, problem = self.current_patient_grouping(
            all_fnames)
        if problem:
            QtWidgets.QMessageBox.about(self, 'Project Creation Error', problem)
            return
        if patient_of is None:
            # Per-image split, as in projects made before patient grouping.
            split_method = f'{order_name}_5_train_to_1_val_v1'
            train_fnames, val_fnames = fixed_train_val_split(all_fnames)
            patient_groups = None
        else:
            split_method = f'patient_grouped_{order_name}_5_train_to_1_val_v1'
            train_fnames, val_fnames = patient_grouped_train_val_split(
                all_fnames, patient_of)
            patient_groups = patient_group_numbers(all_fnames, patient_of)

        os.makedirs(self.sync_dir / project_location)
        proj_file_path = (self.sync_dir / project_location /
                          (project_name + '.seg_proj'))
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'train')
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'val')
        os.makedirs(self.sync_dir / project_location / 'segmentations')
        os.makedirs(self.sync_dir / project_location / 'models')
        os.makedirs(self.sync_dir / project_location / 'messages')
        os.makedirs(self.sync_dir / project_location / 'logs')
        os.makedirs(self.sync_dir / project_location / 'reviews')
        os.makedirs(self.sync_dir / project_location / 'review_exports')

        if self.use_random_weights:
            original_model_file = 'random weights'
        else:
            model_num = 1
            model_name = str(model_num).zfill(6)
            model_name += '_' + str(int(round(time.time()))) + '.pkl'
            shutil.copyfile(self.selected_model,
                            self.sync_dir / project_location /
                            'models' / model_name)
            original_model_file = self.selected_model

        dataset_abs_path = os.path.abspath(dataset_path)
        datasets_abs_path = os.path.abspath(os.path.join(self.sync_dir, 'datasets'))
        # remove the sync_dir/datasets part from the initial part of the dataset path.
        # as the server will prepend the 'datasets' directory when searching for the dataset.
        dataset_rel_path = os.path.relpath(dataset_abs_path, datasets_abs_path)
        manifest_rel_path = None
        if manifest_path is not None:
            manifest_rel_path = os.path.relpath(
                os.path.abspath(manifest_path), datasets_abs_path)

        # create project file.
        project_info = {
            'name': project_name,
            'dataset': str(PurePosixPath(dataset_rel_path)),
            'original_model_file': original_model_file,
            'location': str(PurePosixPath(project_location)),
            'file_names': all_fnames,
            'image_order_method': image_order_method,
            'dataset_manifest': (
                str(PurePosixPath(manifest_rel_path))
                if manifest_rel_path is not None else None),
            'image_order_seed': self.image_order_seed,
            'training_seed': self.image_order_seed,
            'train_file_names': train_fnames,
            'val_file_names': val_fnames,
            'split_method': split_method,
            # How each image's patient was found, and one opaque patient
            # number per entry of file_names (None for a per-image split).
            'patient_rule': patient_rule,
            'patient_groups': patient_groups,
            'model_type': self.model_type,
            'review_schema_version': 1,
            'auto_start_training_after_reviews': 10,
            # A completed "RIPL absent" review supervises the whole scan as
            # background (off: only painted pixels are supervised).
            'absent_scans_as_background': (
                self.absent_background_checkbox.isChecked())
        }
        # 'classes': self.palette_edit_widget.get_brush_data()
        with open(proj_file_path, 'w') as json_file:
            json.dump(project_info, json_file, indent=4)
        self.created.emit(proj_file_path)
        self.close()
