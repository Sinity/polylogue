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
                "pytest.skip('feature not available on this platform')",
                "pytest.xfail('tracked in #1062')",
                "value = dynamic()  # type: ignore[no-any-return]",
                "def unused(arg):  # noqa: ARG001",
                "    pass",
                "def fallback():  # pragma: no cover - defensive",
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
    assert data["discipline_violations_total"] == 0
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
    source.write_text(
        "import pytest\n@pytest.mark.xfail(reason='tracked in #1062', strict=True)\ndef test_x(): ...\n",
        encoding="utf-8",
    )

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
    source.write_text(
        "import pytest\n@pytest.mark.xfail(reason='tracked in #1062', strict=True)\ndef test_x(): ...\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--enforce-discovered"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out
    assert "unregistered source suppressions" not in captured.out


def test_pytest_suppression_scan_ignores_ast_parse_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_parse(_text: str) -> object:
        raise RecursionError("synthetic parser recursion")

    monkeypatch.setattr("devtools.verify_suppressions.ast.parse", fail_parse)

    assert verify_suppressions._pytest_suppression_lines("pytest.mark.xfail(strict=True, reason='ref #1')") == []


def test_skip_without_reason_is_discipline_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "tests" / "test_bad_skip.py"
    source.parent.mkdir()
    source.write_text(
        "import pytest\npytest.skip()\npytest.skip('TODO')\npytest.skip('No claude-code messages')\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert data["blocking"] is True
    # Two of three skips should fail discipline (empty and 'TODO'); the
    # substantive third must pass.
    assert data["discipline_violations_total"] == 2
    for violation in data["discipline_violations"]:
        assert violation["kind"] == "pytest_skip"
        assert any("substantive reason" in err for err in violation["discipline_errors"])


def test_xfail_without_issue_link_is_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "tests" / "test_bad_xfail.py"
    source.parent.mkdir()
    source.write_text(
        "import pytest\n"
        "@pytest.mark.xfail(reason='broken', strict=True)\n"
        "def test_a(): pass\n"
        "@pytest.mark.xfail(reason='tracked in #1062', strict=True)\n"
        "def test_b(): pass\n"
        "@pytest.mark.xfail(reason='tracked in #1062')\n"  # missing strict=True
        "def test_c(): pass\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert data["discipline_violations_total"] == 2
    error_texts = [err for v in data["discipline_violations"] for err in v["discipline_errors"]]
    assert any("must reference an issue" in err for err in error_texts)
    assert any("strict=True" in err for err in error_texts)


def test_bare_type_ignore_comment_is_violation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "polylogue" / "module.py"
    source.parent.mkdir()
    source.write_text(
        "x = 1  # type: ignore\ny = 2  # type: ignore[attr-defined]\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert data["discipline_violations_total"] == 1
    violation = data["discipline_violations"][0]
    assert violation["kind"] == "type_ignore"
    assert any("bare '# type: ignore'" in err for err in violation["discipline_errors"])


def test_no_cover_pragma_requires_justification(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "polylogue" / "module.py"
    source.parent.mkdir()
    source.write_text(
        "def a():  # pragma: no cover\n    pass\n"
        "def b():  # pragma: no cover - defensive\n    pass\n"
        "def c():  # pragma: no cover \u2014 returns RootModeRequest\n    pass\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert data["discipline_violations_total"] == 1
    violation = data["discipline_violations"][0]
    assert violation["kind"] == "no_cover"
    assert any("inline justification" in err for err in violation["discipline_errors"])


def test_skip_with_fstring_reason_is_substantive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    yaml = _write_registry(tmp_path, "suppressions: []\n")
    source = tmp_path / "tests" / "test_fstring_skip.py"
    source.parent.mkdir()
    source.write_text(
        "import pytest\nprovider='codex'\npytest.skip(f'No {provider} messages in seeded database')\n",
        encoding="utf-8",
    )

    rc = verify_suppressions.main(["--yaml", str(yaml), "--scan-root", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert data["discipline_violations_total"] == 0


def test_committed_codebase_passes_discipline(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed codebase must satisfy every discipline rule.

    This regression test is the one place that fails fast when a new
    skip, xfail, no-cover pragma, or bare type-ignore lands without the
    required metadata. Pack B's burn-down work is what makes it pass.
    """
    rc = verify_suppressions.main(["--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0, f"committed codebase has discipline violations: {data['discipline_violations']}"
    assert data["discipline_violations_total"] == 0
