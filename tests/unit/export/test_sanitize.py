"""Unit + fail-closed gate tests for sanitized export (#2381).

The security contract is: no private absolute path and no known-secret pattern
may survive into the written bundle, and the writer must refuse (fail-closed) if
the independent gate finds a leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.export.sanitize import (
    REDACTED_PATH_PREFIX,
    REDACTED_SECRET,
    SanitizedExportError,
    SanitizedExportRequest,
    produce_sanitized_export,
    sanitize_rows,
    verify_sanitized_export,
    write_sanitized_bundle,
)
from polylogue.schemas.privacy_config import PrivacyConfig
from polylogue.schemas.redaction_report import SchemaReport

PLANTED_ABS_PATH = "/home/someone/secret/project/notes.md"
PLANTED_HOME = "/home/someone"
PLANTED_HOME_PATH = "/home/someone/private/keys.txt"
PLANTED_SECRET = "sk-ABCD1234EFGH5678IJKL"


def _planted_rows() -> list[dict[str, object]]:
    return [
        {
            "id": "session-abc-123",
            "display_title": f"work in {PLANTED_ABS_PATH}",
            "origin": "claude-code-session",
            "repo_names": ["polylogue", PLANTED_HOME_PATH],
            "token": PLANTED_SECRET,
            "message_count": 12,
        }
    ]


def test_sanitize_rows_redacts_path_home_and_secret() -> None:
    rows = _planted_rows()
    out, report = sanitize_rows(rows, config=PrivacyConfig(level="standard"))

    blob = json.dumps(out)
    assert PLANTED_ABS_PATH not in blob
    assert PLANTED_HOME_PATH not in blob
    assert PLANTED_SECRET not in blob
    assert REDACTED_PATH_PREFIX in out[0]["display_title"]  # type: ignore[operator]
    assert out[0]["token"] == REDACTED_SECRET
    assert PLANTED_HOME not in blob

    # Every planted leak is recorded as a rejected decision with a reason.
    assert report.total_rejected >= 3
    reasons = set(report.rejection_reasons)
    assert "absolute_path" in reasons
    assert "secret_pattern" in reasons or "high_entropy_token" in reasons
    # The manifest itself must never store the original secret/path verbatim.
    manifest_blob = json.dumps(report.to_json())
    assert PLANTED_SECRET not in manifest_blob
    assert PLANTED_ABS_PATH not in manifest_blob


def test_sanitize_rows_keeps_structural_and_numeric_fields() -> None:
    rows = _planted_rows()
    out, _ = sanitize_rows(rows, config=PrivacyConfig(level="standard"))
    assert out[0]["id"] == "session-abc-123"
    assert out[0]["origin"] == "claude-code-session"
    assert out[0]["message_count"] == 12
    # The benign repo name is preserved; only the path-shaped one is redacted.
    repo_names = out[0]["repo_names"]
    assert isinstance(repo_names, list)
    assert "polylogue" in repo_names


def test_verify_gate_flags_written_leaks(tmp_path: Path) -> None:
    leaky = tmp_path / "bundle"
    leaky.mkdir()
    (leaky / "dataset.jsonl").write_text(
        json.dumps({"display_title": f"see {PLANTED_ABS_PATH}", "token": PLANTED_SECRET}) + "\n",
        encoding="utf-8",
    )
    result = verify_sanitized_export(leaky, home=PLANTED_HOME)
    assert result.ok is False
    assert result.absolute_path_leaks
    assert result.secret_leaks
    assert result.home_path_leaks


def test_verify_gate_passes_clean_bundle(tmp_path: Path) -> None:
    clean = tmp_path / "bundle"
    clean.mkdir()
    (clean / "dataset.jsonl").write_text(
        json.dumps({"display_title": f"{REDACTED_PATH_PREFIX}/notes.md", "token": REDACTED_SECRET}) + "\n",
        encoding="utf-8",
    )
    result = verify_sanitized_export(clean, home=PLANTED_HOME)
    assert result.ok is True
    assert result.absolute_path_leaks == ()
    assert result.secret_leaks == ()


def test_write_refuses_and_cleans_up_on_leak(tmp_path: Path) -> None:
    """Fail-closed: incomplete redaction → gate refuses → no output written."""
    out_dir = tmp_path / "share"
    # Simulate incomplete redaction: the row still carries a raw absolute path.
    leaky_rows: list[dict[str, object]] = [{"display_title": f"oops {PLANTED_ABS_PATH}"}]
    request = SanitizedExportRequest(output_path=out_dir)
    empty_report = SchemaReport(provider="sanitized_export")

    with pytest.raises(SanitizedExportError):
        write_sanitized_bundle(
            rows=leaky_rows,
            report=empty_report,
            scope={},
            request=request,
            home=PLANTED_HOME,
            run_gate=True,
        )

    # Nothing published and no temp dir left behind.
    assert not out_dir.exists()
    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".share.tmp-")]
    assert leftover == []


def test_no_redact_refused_without_acknowledgement(tmp_path: Path) -> None:
    request = SanitizedExportRequest(output_path=tmp_path / "raw", redact=False, acknowledge_unredacted=False)
    with pytest.raises(SanitizedExportError):
        produce_sanitized_export(rows=_planted_rows(), scope={}, request=request)
    assert not (tmp_path / "raw").exists()


def test_produce_writes_gated_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "share"
    request = SanitizedExportRequest(output_path=out_dir)
    result = produce_sanitized_export(
        rows=_planted_rows(), scope={"query": "repo:x"}, request=request, home=PLANTED_HOME
    )

    assert result.verify_ok is True
    assert result.row_count == 1
    dataset_text = result.dataset_path.read_text(encoding="utf-8")
    manifest_text = result.manifest_path.read_text(encoding="utf-8")
    for leak in (PLANTED_ABS_PATH, PLANTED_HOME_PATH, PLANTED_SECRET):
        assert leak not in dataset_text
        assert leak not in manifest_text
    assert result.readme_path.exists()


def test_space_containing_path_is_fully_redacted() -> None:
    """A path with spaces must not leak its directory structure (#2381).

    Regression: a space-naive path regex redacted ``/home/me/`` but left
    ``Secret Project/notes.md`` exposed, and the gate (same pattern) did not
    catch it. The sensitive directory name must not survive.
    """
    rows = [{"title": "edited /home/me/Secret Project/plans.md today"}]
    sanitized, report = sanitize_rows(rows, config=PrivacyConfig(level="standard"))
    title = sanitized[0]["title"]
    assert isinstance(title, str)
    assert "Secret Project" not in title
    assert "/home/me" not in title
    assert report.rejection_reasons.get("absolute_path", 0) >= 1


def test_gate_refuses_raw_space_containing_path(tmp_path: Path) -> None:
    """The fail-closed gate must flag a raw space-containing absolute path."""
    (tmp_path / "dataset.jsonl").write_text(
        json.dumps({"x": "/home/me/Secret Project/notes.md"}) + "\n",
        encoding="utf-8",
    )
    verdict = verify_sanitized_export(tmp_path, home="/home/me")
    assert verdict.ok is False
    assert verdict.absolute_path_leaks


def test_single_segment_root_path_is_redacted_and_gated() -> None:
    """Single-segment system roots (``/etc``) must not survive (#2381 review)."""
    rows = [{"title": "ran in /etc and /tmp", "ok": "/etcetera stays"}]
    sanitized, _ = sanitize_rows(rows, config=PrivacyConfig(level="standard"))
    assert "/etc" not in sanitized[0]["title"]  # type: ignore[operator]
    assert "/tmp" not in sanitized[0]["title"]  # type: ignore[operator]
    assert "/etcetera" in sanitized[0]["ok"]  # type: ignore[operator]


def test_gate_independently_catches_raw_root_and_tilde(tmp_path: Path) -> None:
    """The authoritative gate flags raw single-segment roots and ``~/`` paths."""
    (tmp_path / "dataset.jsonl").write_text(
        json.dumps({"x": "/etc", "y": "~/private/key.txt"}) + "\n",
        encoding="utf-8",
    )
    verdict = verify_sanitized_export(tmp_path, home="/nonexistent-home")
    assert verdict.ok is False
    assert verdict.absolute_path_leaks
    assert verdict.home_path_leaks
