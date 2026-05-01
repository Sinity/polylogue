"""Tests for ``devtools verify-file-budgets``."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_file_budgets


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "budgets.yaml"
    p.write_text(content)
    return p


def test_parse_yaml_round_trip(tmp_path: Path) -> None:
    yaml = _write_yaml(
        tmp_path,
        """defaults:
  source_loc_ceiling: 800
  test_loc_ceiling: 1200

per_package:
  devtools/:
    source_loc_ceiling: 1500

exceptions:
  - path: tests/unit/sources/test_source_laws.py
    ceiling: 2500
    reason: split source/parser/provider contracts
""",
    )
    parsed = verify_file_budgets.parse_yaml(yaml.read_text())
    assert parsed["defaults"] == {"source_loc_ceiling": 800, "test_loc_ceiling": 1200}
    assert parsed["per_package"] == {"devtools/": {"source_loc_ceiling": 1500}}
    assert parsed["exceptions"][0]["path"] == "tests/unit/sources/test_source_laws.py"
    assert parsed["exceptions"][0]["ceiling"] == 2500
    assert parsed["exceptions"][0]["reason"] == "split source/parser/provider contracts"


def test_budget_for_resolution_order() -> None:
    budgets = {
        "defaults": {"source_loc_ceiling": 800, "test_loc_ceiling": 1200},
        "per_package": {"devtools/": {"source_loc_ceiling": 1500}},
        "exceptions": [
            {
                "path": "tests/unit/sources/test_source_laws.py",
                "ceiling": 2500,
                "reason": "split source/parser/provider contracts",
            },
        ],
    }
    # Exception wins.
    ceiling, source = verify_file_budgets.budget_for("tests/unit/sources/test_source_laws.py", budgets)
    assert ceiling == 2500
    assert "exception" in source

    # Per-package wins over default.
    ceiling, source = verify_file_budgets.budget_for("devtools/foo.py", budgets)
    assert ceiling == 1500
    assert "per_package" in source

    # Default fallback.
    ceiling, source = verify_file_budgets.budget_for("polylogue/foo.py", budgets)
    assert ceiling == 800
    assert source == "defaults.source_loc_ceiling"

    ceiling, source = verify_file_budgets.budget_for("tests/unit/foo/test_x.py", budgets)
    assert ceiling == 1200
    assert source == "defaults.test_loc_ceiling"


def test_committed_budgets_are_clean(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed budgets file should pass against the realized tree."""
    rc = verify_file_budgets.main([])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "blocking=False" in captured.out
