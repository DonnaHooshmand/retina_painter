"""Pytest entry points for the New Project patient-control checks.

The checks live in ``smoke_create_project_patients.py`` so they can also be
run with an environment that has PyQt5 but no pytest.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PyQt5")
sys.path.insert(0, str(Path(__file__).parent))
import smoke_create_project_patients as checks  # noqa: E402


@pytest.mark.parametrize(
    'check', checks.CHECKS, ids=[check.__name__ for check in checks.CHECKS])
def test_new_project_patient_controls(check, tmp_path, monkeypatch):
    dialogs = []
    monkeypatch.setattr(
        checks.QtWidgets.QMessageBox, 'about',
        lambda _parent, title, text, *rest: dialogs.append((title, text)))

    check(tmp_path)

    assert not dialogs
