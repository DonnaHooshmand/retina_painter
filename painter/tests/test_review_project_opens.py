"""Pytest entry point for the error-review project check.

The check lives in ``smoke_review_project.py`` so it can also be run with an
environment that has PyQt5 but no pytest.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PyQt5")
sys.path.insert(0, str(Path(__file__).parent))
import smoke_review_project as smoke  # noqa: E402


def test_review_project_opens_steps_and_never_trains(tmp_path):
    with smoke.patched_painter() as dialogs:
        smoke.check_review_project_opens_steps_and_never_trains(
            tmp_path, dialogs)
