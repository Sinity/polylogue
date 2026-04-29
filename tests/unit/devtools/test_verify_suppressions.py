"""Tests for ``devtools verify-suppressions``."""

from __future__ import annotations

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
    import json

    data = json.loads(captured.out)
    assert data["blocking"] is False
    assert data["total"] == 1


def test_committed_registry_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed suppressions.yaml should pass as-is (empty is valid)."""
    rc = verify_suppressions.main([])
    assert rc == 0
    captured = capsys.readouterr()
    assert "blocking=False" in captured.out
