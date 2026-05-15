"""Tests for ``devtools verify-suppressions``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_suppressions


def _write_registry(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "suppressions.yaml"
    p.write_text(content)
    return p


def test_all_current(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(
        tmp_path,
        """suppressions:
  - id: my-sup
    reason: temporary
    expires_at: "2099-12-31"
    issue: "#1"
""",
    )
    rc = verify_suppressions.main(["--yaml", str(yaml)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out


def test_expired_blocking(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(
        tmp_path,
        """suppressions:
  - id: stale-sup
    reason: should have been removed
    expires_at: "2025-01-01"
    issue: "#1"
""",
    )
    rc = verify_suppressions.main(["--yaml", str(yaml)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "blocking=True" in captured.out
    assert "[BLOCK]" in captured.out
    assert "stale-sup" in captured.out


def test_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(
        tmp_path,
        """suppressions:
  - id: active
    reason: ok
    expires_at: "2099-12-31"
""",
    )
    rc = verify_suppressions.main(["--yaml", str(yaml), "--json"])
    assert rc == 0
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["blocking"] is False
    assert data["total"] == 1


def test_committed_registry_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed suppressions.yaml should pass as-is (empty is valid)."""
    rc = verify_suppressions.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out


def test_discovers_source_suppressions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "polylogue" / "example.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                "import pytest",
                "pytest.skip('not available')",
                "pytest.xfail('tracked elsewhere')",
                "value = dynamic()  # type: ignore[no-any-return]",
                "def unused(arg):  # noqa: ARG001",
                "    pass",
                "def fallback():  # pragma: no cover",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])

    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["blocking"] is False
    assert data["discovered_total"] == 5
    assert data["unregistered_total"] == 5
    assert data["discovered_by_kind"] == {
        "no_cover": 1,
        "noqa": 1,
        "pytest_skip": 1,
        "pytest_xfail": 1,
        "type_ignore": 1,
    }


def test_enforce_discovered_blocks_unregistered(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "tests" / "test_example.py"
    source.parent.mkdir()
    source.write_text("@pytest.mark.xfail(reason='tracked elsewhere')\ndef test_x(): ...\n", encoding="utf-8")

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--enforce-discovered"])

    assert rc == 1
    captured = capsys.readouterr()
    assert "[BLOCK] unregistered source suppressions: 1" in captured.out
    assert "blocking=True" in captured.out


def test_enforce_discovered_accepts_registered_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(
        tmp_path,
        """suppressions:
  - id: tracked-exception
    reason: tracked while the replacement lands
    expires_at: "2099-12-31"
    issue: "#1062"
    paths:
      - "tests/test_example.py"
""",
    )
    source = tmp_path / "tests" / "test_example.py"
    source.parent.mkdir()
    source.write_text("@pytest.mark.xfail(reason='tracked elsewhere')\ndef test_x(): ...\n", encoding="utf-8")

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--enforce-discovered"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out
    assert "unregistered source suppressions" not in captured.out


def test_pytest_suppression_scan_ignores_ast_parse_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_parse(_text: str) -> object:
        raise RecursionError("synthetic parser recursion")

    monkeypatch.setattr("devtools.verify_suppressions.ast.parse", fail_parse)

    assert verify_suppressions._pytest_suppression_lines("pytest.mark.xfail()") == []
