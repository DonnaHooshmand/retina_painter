"""Source trainer discovery uses the native venv and preserves launch priority."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "main" / "python"))

import server_manager  # noqa: E402


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    # Spaces in both repository and sync-directory names must stay single args.
    root = tmp_path / "source checkout"
    (root / "painter").mkdir(parents=True)
    _touch(root / "trainer" / "src" / "main.py")
    monkeypatch.setattr(server_manager, "_find_repo_root_from_here", lambda: root)
    monkeypatch.delenv("ROOTPAINTER_TRAINER", raising=False)
    return root


@pytest.mark.parametrize("platform, interpreter", [
    ("win32", "Scripts/python.exe"),
    ("linux", "bin/python"),
    ("darwin", "bin/python"),
])
def test_native_source_venv_and_launch_arguments(checkout, monkeypatch, platform, interpreter):
    monkeypatch.setattr(server_manager.sys, "platform", platform)
    expected_python = _touch(checkout / "trainer" / "env" / interpreter)
    spec = server_manager.find_dev_trainer_launch()

    assert spec.program == expected_python
    assert spec.args_prefix == ["-u", str(checkout / "trainer" / "src" / "main.py"), "--syncdir"]
    assert spec.working_dir == checkout / "trainer" / "src"
    sync_dir = checkout / "separate synthetic sync"
    assert (spec.args_prefix + [str(sync_dir)])[-1] == str(sync_dir)


@pytest.mark.parametrize("platform, foreign_interpreter", [
    ("win32", "bin/python"),
    ("linux", "Scripts/python.exe"),
])
def test_foreign_venv_is_not_a_working_trainer(checkout, monkeypatch, platform, foreign_interpreter):
    monkeypatch.setattr(server_manager.sys, "platform", platform)
    _touch(checkout / "trainer" / "env" / foreign_interpreter)
    assert server_manager.find_dev_trainer_launch() is None


def test_missing_entry_point_is_not_launchable(checkout, monkeypatch):
    monkeypatch.setattr(server_manager.sys, "platform", "win32")
    _touch(checkout / "trainer" / "env" / "Scripts" / "python.exe")
    (checkout / "trainer" / "src" / "main.py").unlink()
    assert server_manager.find_dev_trainer_launch() is None


def test_missing_checkout_is_not_launchable(monkeypatch):
    monkeypatch.setattr(server_manager, "_find_repo_root_from_here", lambda: None)
    assert server_manager.find_dev_trainer_launch() is None


def test_explicit_executable_precedes_bundle_and_dev(checkout, monkeypatch):
    override = _touch(checkout / "custom trainer.exe")
    monkeypatch.setenv("ROOTPAINTER_TRAINER", str(override))
    monkeypatch.setattr(server_manager, "find_bundled_trainer", lambda: pytest.fail("Override ignored"))
    spec = server_manager.find_trainer_launch()
    assert spec.program == override
    assert spec.args_prefix == ["--syncdir"]


def test_bundle_precedes_dev(checkout, monkeypatch):
    bundled = _touch(checkout / "RetinaPainterTrainer.exe")
    monkeypatch.setattr(server_manager, "find_bundled_trainer", lambda: bundled)
    monkeypatch.setattr(server_manager, "find_dev_trainer_launch", lambda: pytest.fail("Bundle ignored"))
    assert server_manager.find_trainer_launch().program == bundled


def test_invalid_override_falls_back_to_native_dev(checkout, monkeypatch):
    monkeypatch.setattr(server_manager.sys, "platform", "win32")
    monkeypatch.setenv("ROOTPAINTER_TRAINER", str(checkout / "missing.exe"))
    monkeypatch.setattr(server_manager, "find_bundled_trainer", lambda: None)
    expected_python = _touch(checkout / "trainer" / "env" / "Scripts" / "python.exe")
    assert server_manager.find_trainer_launch().program == expected_python
