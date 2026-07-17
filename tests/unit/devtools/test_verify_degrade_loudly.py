"""Tests for the degrade-loudly review-gate lint (polylogue-cpf.4).

The lint is the "review-gate or lint flags new bare soft-fails" half of the
bead's acceptance criteria — these tests prove it actually catches a new
silent soft-fail, accepts a logged/allowlisted one, and rejects a stale
allowlist entry, using synthetic fixture trees (not the real repo, so a
future edit to real source can't accidentally make this test vacuous).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from devtools import verify_degrade_loudly


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _run_json(root: Path, capsys: pytest.CaptureFixture[str], *, allowlist: Path | None = None) -> dict[str, Any]:
    args = ["--json", "--root", str(root)]
    if allowlist is not None:
        args += ["--allowlist", str(allowlist)]
    rc = verify_degrade_loudly.main(args)
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    payload["_rc"] = rc
    return payload


def test_flags_new_silent_except_with_no_log_or_signal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A bare ``except Exception: return None`` with no log call is exactly
    the pattern polylogue-cpf.4 targets."""
    _write(
        tmp_path,
        "polylogue/daemon/example_status.py",
        """
        def get_widget_count(path):
            try:
                return _query(path)
            except Exception:
                return None
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    violations = payload["violations"]
    assert len(violations) == 1
    assert violations[0]["path"] == "polylogue/daemon/example_status.py"
    assert violations[0]["function"] == "<module>.get_widget_count"
    assert violations[0]["exceptions"] == ["Exception"]


def test_accepts_logged_except_with_no_allowlist_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Adding a log call is the cheapest fix and needs no allowlist entry."""
    _write(
        tmp_path,
        "polylogue/daemon/example_status.py",
        """
        def get_widget_count(path):
            try:
                return _query(path)
            except Exception as exc:
                logger.warning("widget count query failed: %s", exc, exc_info=True)
                return None
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["ok"] is True
    assert payload["violations"] == []


def test_accepts_reraise_with_no_allowlist_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/storage/example_repair.py",
        """
        def repair(path):
            try:
                return _run(path)
            except Exception as exc:
                raise RuntimeError("repair failed") from exc
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["violations"] == []


def test_narrow_exceptions_are_not_flagged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """ValueError/TypeError/JSONDecodeError-style narrow coercion is out of
    scope -- only broad Exception/BaseException/*.Error catches are flagged."""
    _write(
        tmp_path,
        "polylogue/storage/example_mappers.py",
        """
        def coerce_int(value, default):
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["violations"] == []


def test_allowlisted_site_passes_with_matching_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "polylogue/insights/example_audit.py",
        """
        def build_report(path):
            try:
                return _query(path)
            except Exception as exc:
                return {"available": False, "error": str(exc)}
        """,
    )
    allowlist = tmp_path / "docs" / "plans" / "degrade-loudly-allowlist.yaml"
    _write(
        tmp_path,
        "docs/plans/degrade-loudly-allowlist.yaml",
        """
        entries:
        - path: polylogue/insights/example_audit.py
          function: <module>.build_report
          exceptions: [Exception]
          occurrence: 0
          reason: 'Returns {"available": False, "error": str(exc)} -- already typed.'
        """,
    )

    payload = _run_json(tmp_path, capsys, allowlist=allowlist)

    assert payload["_rc"] == 0
    assert payload["ok"] is True
    assert payload["allowlisted"] == 1
    assert payload["violations"] == []
    assert payload["stale_allowlist_entries"] == []


def test_stale_allowlist_entry_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An allowlist entry with no matching site (the except was removed, or
    logging was added) must fail the gate too -- otherwise the allowlist
    only ever grows and stops meaning anything."""
    _write(
        tmp_path,
        "polylogue/insights/example_audit.py",
        """
        def build_report(path):
            return _query(path)
        """,
    )
    allowlist = tmp_path / "docs" / "plans" / "degrade-loudly-allowlist.yaml"
    _write(
        tmp_path,
        "docs/plans/degrade-loudly-allowlist.yaml",
        """
        entries:
        - path: polylogue/insights/example_audit.py
          function: <module>.build_report
          exceptions: [Exception]
          occurrence: 0
          reason: 'No longer matches any except-handler in this function.'
        """,
    )

    payload = _run_json(tmp_path, capsys, allowlist=allowlist)

    assert payload["_rc"] == 1
    assert payload["ok"] is False
    assert len(payload["stale_allowlist_entries"]) == 1
    assert payload["stale_allowlist_entries"][0]["function"] == "<module>.build_report"


def test_test_files_and_non_target_packages_are_excluded(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write(
        tmp_path,
        "tests/unit/daemon/test_example.py",
        """
        def test_thing():
            try:
                pass
            except Exception:
                pass
        """,
    )
    _write(
        tmp_path,
        "polylogue/cli/example_click.py",
        """
        def run():
            try:
                pass
            except Exception:
                pass
        """,
    )

    payload = _run_json(tmp_path, capsys)

    assert payload["_rc"] == 0
    assert payload["sites_scanned"] == 0


def test_real_repo_allowlist_is_internally_consistent(capsys: pytest.CaptureFixture[str]) -> None:
    """The committed docs/plans/degrade-loudly-allowlist.yaml must currently
    match the real repo exactly -- no unallowlisted sites, no stale entries.
    This is the gate that runs in ``devtools verify``."""
    assert verify_degrade_loudly.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []
    assert payload["stale_allowlist_entries"] == []
