"""Tests for ``devtools verify-migrations``."""

from __future__ import annotations

from pathlib import Path

import pytest

from devtools import verify_migrations


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "migrations.yaml"
    p.write_text(content)
    return p


def test_parse_yaml_one_migration(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """migrations:
  retire-foo:
    issue: "#100"
    description: "Test migration."
    must_vanish_paths:
      - polylogue/foo.py
      - tests/baselines/foo.txt
    must_vanish_cli_commands:
      - foo
    must_vanish_devtools_commands: []
    forbidden_substrings:
      - location_glob: "polylogue/**/*.py"
        substring: "foo_legacy"

completed: []
""",
    )
    parsed = verify_migrations.parse_yaml(yaml.read_text())
    spec = parsed["migrations"]["retire-foo"]
    assert spec["issue"] == "#100"
    assert spec["must_vanish_paths"] == ["polylogue/foo.py", "tests/baselines/foo.txt"]
    assert spec["must_vanish_cli_commands"] == ["foo"]
    assert spec["forbidden_substrings"][0]["substring"] == "foo_legacy"


def test_parse_yaml_empty_active_manifest(tmp_path: Path) -> None:
    yaml = _write(
        tmp_path,
        """migrations: {}

completed: []
""",
    )

    parsed = verify_migrations.parse_yaml(yaml.read_text())

    assert parsed["migrations"] == {}


def test_check_migration_no_paths_complete(tmp_path: Path) -> None:
    spec = {
        "issue": "#1",
        "must_vanish_paths": [],
        "must_vanish_cli_commands": [],
        "must_vanish_devtools_commands": [],
        "forbidden_substrings": [],
    }
    result = verify_migrations.check_migration("empty", spec)
    # No constraints declared → not "complete" (nothing to check)
    assert not result["complete"]
    assert result["findings"] == {}


def test_check_migration_surviving_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    surviving = tmp_path / "survivor.py"
    surviving.write_text("# still here")
    monkeypatch.setattr(verify_migrations, "ROOT", tmp_path)
    spec = {
        "issue": "#X",
        "must_vanish_paths": ["survivor.py", "absent.py"],
        "must_vanish_cli_commands": [],
        "must_vanish_devtools_commands": [],
    }
    result = verify_migrations.check_migration("test", spec)
    assert "survivor.py" in result["findings"]["surviving_paths"]
    assert "absent.py" not in result["findings"].get("surviving_paths", [])


def test_committed_manifest_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed manifest should parse and report (informational, not blocking)."""
    rc = verify_migrations.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out


def test_json_unknown_strict_name_keeps_stdout_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    rc = verify_migrations.main(["--json", "--strict", "missing-migration"])

    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out.startswith("{")
    assert "[error]" not in captured.out
    assert "[error]" in captured.err
