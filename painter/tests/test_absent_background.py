"""Pytest entry points for the absent-scans-as-background checks.

The checks live in ``smoke_absent_background.py`` so they can also be run with
an environment that has PyQt5 but no pytest.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
pytest.importorskip("PyQt5")
sys.path.insert(0, str(Path(__file__).parent))
import smoke_absent_background as checks  # noqa: E402


@pytest.mark.parametrize(
    'check', checks.CHECKS, ids=[check.__name__ for check in checks.CHECKS])
def test_absent_scans_as_background(check, tmp_path):
    with checks.patched_painter() as dialogs:
        check(tmp_path, dialogs)
