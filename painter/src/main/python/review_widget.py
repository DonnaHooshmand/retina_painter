"""Scan-level clinician review controls."""

from PyQt5 import QtCore, QtWidgets

from review_utils import VALID_DECISIONS


class ReviewWidget(QtWidgets.QWidget):
    """Collect the explicit scan decision and completion state."""

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.setLayout(layout)

        layout.addWidget(QtWidgets.QLabel("Scan decision:"))
        self.decision_combo = QtWidgets.QComboBox()
        self.decision_combo.addItem("Select…", None)
        self.decision_combo.addItem("RIPL present", "present")
        self.decision_combo.addItem("RIPL absent", "absent")
        self.decision_combo.addItem("Uncertain", "uncertain")
        self.decision_combo.setToolTip(
            "The clinician's B-scan-level RIPL decision.")
        self.decision_combo.currentIndexChanged.connect(
            self._update_complete_enabled)
        layout.addWidget(self.decision_combo)

        self.complete_checkbox = QtWidgets.QCheckBox("Review complete")
        self.complete_checkbox.setToolTip(
            "Check only after the entire B-scan and all corrections have "
            "been reviewed.")
        self.complete_checkbox.setEnabled(False)
        layout.addWidget(self.complete_checkbox)

        self.count_label = QtWidgets.QLabel("0 completed")
        layout.addWidget(self.count_label)
        layout.addStretch()

    def _update_complete_enabled(self):
        has_decision = self.decision() in VALID_DECISIONS
        if not has_decision:
            self.complete_checkbox.setChecked(False)
        self.complete_checkbox.setEnabled(has_decision)

    def decision(self):
        return self.decision_combo.currentData()

    def is_complete(self):
        return self.complete_checkbox.isChecked()

    def set_review(self, decision=None, review_complete=False):
        self.decision_combo.blockSignals(True)
        self.complete_checkbox.blockSignals(True)
        index = self.decision_combo.findData(decision)
        self.decision_combo.setCurrentIndex(max(index, 0))
        self.complete_checkbox.setEnabled(decision in VALID_DECISIONS)
        self.complete_checkbox.setChecked(
            bool(review_complete and decision in VALID_DECISIONS))
        self.complete_checkbox.blockSignals(False)
        self.decision_combo.blockSignals(False)

    def set_completed_count(self, count):
        self.count_label.setText(f"{count} completed")
