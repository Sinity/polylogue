"""Tests for ``devtools verify closure-matrix``."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_closure_matrix


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_committed_matrix_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed docs/plans/test-closure-matrix.yaml validates against the tree."""
    rc = verify_closure_matrix.main([])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "blocking=False" in captured.out


def test_missing_target_file_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: phantom.module
    target_files:
      - polylogue/does/not/exist.py
    representative_tests:
      - tests/unit/devtools/test_verify_closure_matrix.py
    gate: required
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "target_files path missing: polylogue/does/not/exist.py" in captured.out


def test_missing_representative_test_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: phantom.test
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests:
      - tests/unit/does/not/exist.py
    gate: required
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "representative_tests path missing" in captured.out


def test_absent_gate_requires_known_gaps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: phantom.absent
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests: []
    gate: absent
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "must list at least one 'known_gaps' bullet" in captured.out


def test_required_gate_requires_representative_tests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: phantom.required
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests: []
    gate: required
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "must list at least one representative test" in captured.out


def test_invalid_gate_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: phantom.gate
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests:
      - tests/unit/devtools/test_verify_closure_matrix.py
    gate: aspirational
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "'gate' must be one of" in captured.out


def test_duplicate_domain_blocks(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: dup.row
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests:
      - tests/unit/devtools/test_verify_closure_matrix.py
    gate: required
  - domain: dup.row
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests:
      - tests/unit/devtools/test_verify_closure_matrix.py
    gate: required
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "duplicate domain" in captured.out


def test_nodeid_form_accepted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml_path = _write(
        tmp_path,
        """rows:
  - domain: nodeid.form
    target_files:
      - devtools/verify_closure_matrix.py
    representative_tests:
      - tests/unit/devtools/test_verify_closure_matrix.py::test_committed_matrix_is_clean
    gate: required
""",
    )
    rc = verify_closure_matrix.main(["--yaml", str(yaml_path)])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
