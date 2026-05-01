"""Tests for ``devtools verify-test-ownership``."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_test_ownership


def test_parse_yaml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "manifest.yaml"
    p.write_text(
        """untested:
  - path: polylogue/__init__.py
    reason: package marker

shared:
  - path: tests/unit/__init__.py
    reason: package marker
""",
    )
    parsed = verify_test_ownership.parse_yaml(p.read_text())
    assert parsed["untested"][0]["path"] == "polylogue/__init__.py"
    assert parsed["shared"][0]["reason"] == "package marker"


def test_imports_in_filters_to_production(tmp_path: Path) -> None:
    test_file = tmp_path / "test_something.py"
    test_file.write_text(
        """import json
import polylogue.core.json as plj
from polylogue.storage import repository
from devtools import verify
from os import path
""",
    )
    imports = verify_test_ownership.imports_in(test_file)
    assert "polylogue.core.json" in imports
    assert "polylogue.storage" in imports
    assert "devtools" in imports
    assert "json" not in imports
    assert "os" not in imports


def test_committed_manifest_is_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed test-ownership manifest should pass."""
    rc = verify_test_ownership.main([])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "blocking=False" in captured.out
