"""The test-packages gate must catch a reintroduced non-package test directory.

Anti-vacuity: delete the gate's ``__init__.py`` check, or drop the ancestor
walk, and ``test_missing_package_directory_is_reported`` /
``test_broken_ancestor_package_chain_is_reported`` go red. Re-create the
collision the gate exists to prevent -- a directory of test modules with no
``__init__.py`` -- and the gate reports it instead of the corpus aborting at
collection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_test_packages import main, missing_packages


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_gate_is_registered_against_its_module() -> None:
    gate = GATES_BY_NAME["test-packages"]
    assert gate.args == ("devtools.verify_test_packages",)
    assert gate.kind == "module"
    assert gate.in_quick is True
    assert gate.blocking is True


def test_fully_packaged_tree_reports_nothing(tmp_path: Path) -> None:
    _write(tmp_path / "__init__.py")
    _write(tmp_path / "unit" / "__init__.py")
    _write(tmp_path / "unit" / "test_alpha.py")

    assert missing_packages(tmp_path) == []


def test_missing_package_directory_is_reported(tmp_path: Path) -> None:
    _write(tmp_path / "__init__.py")
    _write(tmp_path / "unit" / "__init__.py")
    _write(tmp_path / "unit" / "daemon" / "test_status.py")

    assert missing_packages(tmp_path) == [tmp_path / "unit" / "daemon"]


def test_broken_ancestor_package_chain_is_reported(tmp_path: Path) -> None:
    """A leaf package under a non-package parent still imports unqualified."""
    _write(tmp_path / "__init__.py")
    _write(tmp_path / "unit" / "leaf" / "__init__.py")
    _write(tmp_path / "unit" / "leaf" / "test_status.py")

    assert missing_packages(tmp_path) == [tmp_path / "unit"]


def test_conftest_only_directory_must_be_a_package(tmp_path: Path) -> None:
    _write(tmp_path / "__init__.py")
    _write(tmp_path / "helpers" / "conftest.py")

    assert missing_packages(tmp_path) == [tmp_path / "helpers"]


def test_pycache_is_not_a_test_directory(tmp_path: Path) -> None:
    _write(tmp_path / "__init__.py")
    _write(tmp_path / "__pycache__" / "test_alpha.py")

    assert missing_packages(tmp_path) == []


def test_gate_passes_on_this_checkout(capsys: pytest.CaptureFixture[str]) -> None:
    """The real tests/ tree satisfies the invariant the gate declares."""
    assert main([]) == 0
    assert "every directory" in capsys.readouterr().out
