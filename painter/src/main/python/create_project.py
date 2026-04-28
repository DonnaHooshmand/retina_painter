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
import os
import random
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

class CreateProjectWidget(QtWidgets.QWidget):

    created = QtCore.pyqtSignal(Path)

    def __init__(self, sync_dir):
        super().__init__()
        # Single-mode state (5:1 auto-routing — current/default behavior).
        self.selected_dir = None
        # Split-mode state (user pre-organized images into train/ and val/).
        self.split_mode = False
        self.selected_train_dir = None
        self.selected_val_dir = None

        self.proj_name = None
        self.selected_model = None
        self.use_random_weights = True
        self.model_type = 'unet'
        self.sync_dir = sync_dir
        self.initUI()

    def initUI(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.name_edit_widget = NameEditWidget('Project')
        self.name_edit_widget.changed.connect(self.validate)
        self.layout.addWidget(self.name_edit_widget)

        self.add_split_mode_widget()
        self.add_im_dir_widget()
        self.add_split_dir_widget()
        self.add_model_type_widget()
        self.add_radio_widget()
        self.add_model_btn()
        if False:
            self.add_palette_widget()
        self.add_info_label()
        self.add_create_btn()

    def add_split_mode_widget(self):
        """Checkbox to toggle between 5:1 auto-split (default) and explicit
        pre-split train/val folders.

        Auto-split (default) is the original RootPainter behavior: a single
        image directory; the painter routes new annotations between
        annotations/train/ and annotations/val/ using a 5:1 file-count ratio
        with no awareness of patient ID or other grouping.

        Pre-split mode lets the user point at two directories that have
        already been separated (typically by patient ID, to avoid
        within-patient leakage between train and val). The painter will
        then route annotations based on which source folder each image
        came from, rather than by count.
        """
        helper = QtWidgets.QLabel(
            "By default, RetinaPainter routes new annotations between train/ "
            "and val/ using a 5:1 file-count ratio (no awareness of patient "
            "ID). If you've pre-organized your images into separate train "
            "and validation folders (e.g. for patient-level data leakage "
            "prevention), check the box below to use those folders directly."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #555;")
        self.layout.addWidget(helper)

        self.split_mode_checkbox = QtWidgets.QCheckBox(
            "I have pre-split train/val folders"
        )
        self.split_mode_checkbox.toggled.connect(self.on_split_mode_toggled)
        self.layout.addWidget(self.split_mode_checkbox)

    def add_im_dir_widget(self):
        # Single-mode picker (5:1 auto-routing). Wrapped in a container so
        # the whole section can be hidden when the user chooses pre-split mode.
        self.single_dir_container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.single_dir_container.setLayout(layout)

        directory_label = QtWidgets.QLabel()
        directory_label.setText("Image directory: Not yet specified")
        layout.addWidget(directory_label)
        self.directory_label = directory_label

        specify_image_dir_btn = QtWidgets.QPushButton('Specify image directory')
        specify_image_dir_btn.clicked.connect(self.select_photo_dir)
        layout.addWidget(specify_image_dir_btn)

        self.layout.addWidget(self.single_dir_container)

    def add_split_dir_widget(self):
        # Pre-split-mode pickers: one for train, one for val. Hidden by default.
        self.split_dir_container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.split_dir_container.setLayout(layout)

        # Requirements summary so the user sees up front what's expected.
        reqs = QtWidgets.QLabel(
            "Both folders must:\n"
            "  • live inside the sync directory's datasets/ folder\n"
            "  • contain at least one image\n"
            "  • have unique filenames (no image present in both)"
        )
        reqs.setStyleSheet("color: #555;")
        layout.addWidget(reqs)

        # Train folder picker
        self.train_dir_label = QtWidgets.QLabel(
            "Train images directory: Not yet specified"
        )
        layout.addWidget(self.train_dir_label)
        train_btn = QtWidgets.QPushButton('Specify train images directory')
        train_btn.clicked.connect(self.select_train_dir)
        layout.addWidget(train_btn)

        # Val folder picker
        self.val_dir_label = QtWidgets.QLabel(
            "Validation images directory: Not yet specified"
        )
        layout.addWidget(self.val_dir_label)
        val_btn = QtWidgets.QPushButton('Specify validation images directory')
        val_btn.clicked.connect(self.select_val_dir)
        layout.addWidget(val_btn)

        self.split_dir_container.setVisible(False)
        self.layout.addWidget(self.split_dir_container)

    def on_split_mode_toggled(self, checked):
        """Switch between 5:1 auto-split and pre-split modes.

        Clears the path stored for the *other* mode so we don't accidentally
        carry a stale selection across modes.
        """
        self.split_mode = checked
        self.single_dir_container.setVisible(not checked)
        self.split_dir_container.setVisible(checked)
        if checked:
            # Entering pre-split mode — forget any single-mode selection.
            self.selected_dir = None
            self.directory_label.setText("Image directory: Not yet specified")
        else:
            # Leaving pre-split mode — forget train/val selections.
            self.selected_train_dir = None
            self.selected_val_dir = None
            self.train_dir_label.setText(
                "Train images directory: Not yet specified"
            )
            self.val_dir_label.setText(
                "Validation images directory: Not yet specified"
            )
        self.validate()

    def add_model_type_widget(self):
        label = QtWidgets.QLabel("Model type:")
        self.layout.addWidget(label)
        self.model_type_combo = QtWidgets.QComboBox()
        self.model_type_combo.addItem("U-Net (original RootPainter)", "unet")
        self.model_type_combo.addItem("RETFound + plain decoder", "retfound")
        self.model_type_combo.addItem("RETFound + RFA-U-Net (recommended)", "retfound_rfa")
        self.model_type_combo.currentIndexChanged.connect(self.on_model_type_changed)
        self.layout.addWidget(self.model_type_combo)

    def on_model_type_changed(self, index):
        self.model_type = self.model_type_combo.itemData(index)

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

        # Directory validation: branches on auto-split vs pre-split mode.
        if self.split_mode:
            if not self._validate_split_dirs():
                return
        else:
            if not self._validate_single_dir():
                return

        if not self.use_random_weights and not self.selected_model:
            self.info_label.setText("Starting model must be specified to create project")
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
            return

        if self.split_mode:
            # Slice 1: validation passes, but project creation in pre-split
            # mode is wired up in Slice 2. Until then, surface the gating
            # clearly so the user knows what to expect when they click Create.
            self.info_label.setText(
                f"Project location: {self.project_location} — pre-split mode "
                "validation passed. Project creation in pre-split mode is "
                "not yet implemented (Slice 2)."
            )
            self.create_project_btn.setEnabled(True)
        else:
            self.info_label.setText(f"Project location: {self.project_location}")
            self.create_project_btn.setEnabled(True)

    def _validate_single_dir(self) -> bool:
        """Validate the single-directory (auto-split) flow. Returns True if
        the directory check passed; False (and updates info_label + disables
        Create) otherwise."""
        if not self.selected_dir:
            self.info_label.setText("Directory must be specified to create project")
            self.create_project_btn.setEnabled(False)
            return False

        cur_files = os.listdir(self.selected_dir)
        cur_files = [is_image(f) for f in cur_files]
        if not cur_files:
            self.info_label.setText("Folder contains no images.")
            self.create_project_btn.setEnabled(False)
            return False

        return True

    def _validate_split_dirs(self) -> bool:
        """Validate the pre-split (train + val) flow. Returns True if both
        directories are specified, both inside sync_dir/datasets/, both
        contain images, and have no filename overlap."""
        if not self.selected_train_dir:
            self.info_label.setText(
                "Train images directory must be specified."
            )
            self.create_project_btn.setEnabled(False)
            return False
        if not self.selected_val_dir:
            self.info_label.setText(
                "Validation images directory must be specified."
            )
            self.create_project_btn.setEnabled(False)
            return False

        if self.selected_train_dir == self.selected_val_dir:
            self.info_label.setText(
                "Train and validation directories must be different."
            )
            self.create_project_btn.setEnabled(False)
            return False

        # Both must live inside sync_dir/datasets/ — same constraint as
        # the existing single-folder mode (enforced in create_project).
        datasets_dir = os.path.abspath(os.path.join(self.sync_dir, 'datasets'))
        for label, path in (("Train", self.selected_train_dir),
                            ("Validation", self.selected_val_dir)):
            if not os.path.abspath(path).startswith(datasets_dir):
                self.info_label.setText(
                    f"{label} folder must be inside the sync directory's "
                    "datasets/ folder."
                )
                self.create_project_btn.setEnabled(False)
                return False

        train_imgs = [f for f in os.listdir(self.selected_train_dir) if is_image(f)]
        val_imgs = [f for f in os.listdir(self.selected_val_dir) if is_image(f)]
        if not train_imgs:
            self.info_label.setText("Train folder contains no images.")
            self.create_project_btn.setEnabled(False)
            return False
        if not val_imgs:
            self.info_label.setText("Validation folder contains no images.")
            self.create_project_btn.setEnabled(False)
            return False

        # No filename overlap — overlapping filenames mean we can't
        # uniquely route an annotation back to train or val by name.
        overlap = set(train_imgs) & set(val_imgs)
        if overlap:
            sample = sorted(overlap)[:3]
            example = ", ".join(sample) + ("..." if len(overlap) > 3 else "")
            self.info_label.setText(
                f"Filename overlap between train and val ({len(overlap)} "
                f"file(s)): {example}. Filenames must be unique across both "
                "folders."
            )
            self.create_project_btn.setEnabled(False)
            return False

        return True


    def select_photo_dir(self):
        self.photo_dialog = QtWidgets.QFileDialog(self, directory=os.path.join(self.sync_dir, 'datasets'))

        self.photo_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        def output_selected():
            self.selected_dir = self.photo_dialog.selectedFiles()[0]
            self.directory_label.setText('Image directory: ' + self.selected_dir)
            self.validate()

        self.photo_dialog.fileSelected.connect(output_selected)
        self.photo_dialog.open()

    def _open_dataset_dir_dialog(self, on_chosen):
        """Helper for the two split-mode pickers — both pickers want the same
        config (datasets folder default, single-directory selection mode), so
        share one factory rather than duplicating it.
        """
        dialog = QtWidgets.QFileDialog(
            self, directory=os.path.join(self.sync_dir, 'datasets')
        )
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.fileSelected.connect(lambda: on_chosen(dialog.selectedFiles()[0]))
        dialog.open()
        return dialog  # keep a reference alive on the instance to avoid GC

    def select_train_dir(self):
        def chosen(path):
            self.selected_train_dir = path
            self.train_dir_label.setText('Train images directory: ' + path)
            self.validate()
        self._train_dialog = self._open_dataset_dir_dialog(chosen)

    def select_val_dir(self):
        def chosen(path):
            self.selected_val_dir = path
            self.val_dir_label.setText('Validation images directory: ' + path)
            self.validate()
        self._val_dialog = self._open_dataset_dir_dialog(chosen)


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
        # Pre-split mode is not yet wired through to project creation (the
        # downstream painter loading and trainer instruction need their own
        # changes). Surface that clearly rather than silently creating a
        # half-broken project.
        if self.split_mode:
            QtWidgets.QMessageBox.information(
                self, 'Pre-split mode not yet available',
                "The pre-split train/val mode is in development. The "
                "validation here confirms your folders look correct, but "
                "creating a project in this mode is not yet implemented — "
                "that lands in the next slice (Slice 2: project creation, "
                "painter image loading; Slice 3: trainer instruction).\n\n"
                f"Train: {self.selected_train_dir}\n"
                f"Val:   {self.selected_val_dir}\n\n"
                "Untick the box to create a project in the default 5:1 "
                "auto-split mode."
            )
            return

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

        os.makedirs(self.sync_dir / project_location)
        proj_file_path = (self.sync_dir / project_location /
                          (project_name + '.seg_proj'))
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'train')
        os.makedirs(self.sync_dir / project_location / 'annotations' / 'val')
        os.makedirs(self.sync_dir / project_location / 'segmentations')
        os.makedirs(self.sync_dir / project_location / 'models')
        os.makedirs(self.sync_dir / project_location / 'messages')
        os.makedirs(self.sync_dir / project_location / 'logs')

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

        # get files in random order for training.
        all_fnames = file_utils.ls(dataset_path)
        # images only
        all_fnames = [a for a in all_fnames if is_image(a)]

        all_fnames = sorted(all_fnames)
        random.shuffle(all_fnames)
       

        dataset_abs_path = os.path.abspath(dataset_path)
        datasets_abs_path = os.path.abspath(os.path.join(self.sync_dir, 'datasets'))
        # remove the sync_dir/datasets part from the initial part of the dataset path.
        # as the server will prepend the 'datasets' directory when searching for the dataset.
        dataset_rel_path = os.path.relpath(dataset_abs_path, datasets_abs_path)

        # create project file.
        project_info = {
            'name': project_name,
            'dataset': str(PurePosixPath(dataset_rel_path)),
            'original_model_file': original_model_file,
            'location': str(PurePosixPath(project_location)),
            'file_names': all_fnames,
            'model_type': self.model_type
        }
        # 'classes': self.palette_edit_widget.get_brush_data()
        with open(proj_file_path, 'w') as json_file:
            json.dump(project_info, json_file, indent=4)
        self.created.emit(proj_file_path)
        self.close()
