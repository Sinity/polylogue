from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_docs_drift


def test_check_paths_flags_missing_backtick_path() -> None:
    text = "See `polylogue/does/not/exist.py` for details."
    missing = verify_docs_drift._check_paths("fake.md", text)
    assert [m.quoted for m in missing] == ["polylogue/does/not/exist.py"]


def test_check_paths_resolves_package_root_prefix() -> None:
    """A path given relative to the package root (no leading polylogue/) resolves."""
    text = "See `storage/sqlite/archive_tiers/source.py` for the DDL."
    assert verify_docs_drift._check_paths("fake.md", text) == []


def test_check_paths_resolves_tests_prefix() -> None:
    text = "See `infra/corpus_fixtures.py` (relative to tests/) for the fixture."
    assert verify_docs_drift._check_paths("fake.md", text) == []


def test_check_paths_skips_historical_reference() -> None:
    """A path explicitly described as retired/superseded is a historical claim,
    not a live assertion that it exists -- must not be flagged, even when the
    explanatory clause lands on a later line via markdown hard-wrap."""
    text = (
        "the former `docs/execution-plan.md` was\n"
        "superseded (its GitHub-issue map was re-encoded as Beads issues) and retired\n"
        "(Ref polylogue-3tl.13)."
    )
    assert verify_docs_drift._check_paths("fake.md", text) == []


def test_check_paths_skips_cache_and_local_output() -> None:
    """.cache/ and .local/ paths are runtime-generated output, not stale claims."""
    text = "See `.cache/verify/current-pytest-summary.json` for the last run."
    assert verify_docs_drift._check_paths("fake.md", text) == []


def test_check_schema_versions_flags_overclaim() -> None:
    text = "Index schema version 9999 adds a table that does not exist yet."
    current = {"index": 28}
    overclaims = verify_docs_drift._check_schema_versions("fake.md", text, current)
    assert len(overclaims) == 1
    assert overclaims[0].tier == "index"
    assert overclaims[0].claimed == 9999
    assert overclaims[0].current == 28


def test_check_schema_versions_allows_historical_versions() -> None:
    """A version at or below the current constant is a legitimate history entry."""
    text = "Index schema version 24 admits capture_gap rows."
    current = {"index": 28}
    assert verify_docs_drift._check_schema_versions("fake.md", text, current) == []


def test_check_watchlist_flags_bare_mention() -> None:
    text = "Rows are grouped in `artifact_observations` keyed by raw_id."
    hits = verify_docs_drift._check_watchlist("fake.md", text)
    assert [h.term for h in hits] == ["artifact_observations"]


def test_check_watchlist_allows_explained_rename() -> None:
    text = "There is no separate `blob_links` table; `artifact_observations` was renamed to `raw_artifacts`."
    assert verify_docs_drift._check_watchlist("fake.md", text) == []


def test_real_reference_docs_currently_pass() -> None:
    """The actual Reference-docs sweep must be clean after the 9e5.13 fix pass --
    this is the gate the bead's verify clause asks for."""
    report = verify_docs_drift.collect_drift()
    assert report.ok, (report.missing_paths, report.schema_overclaims, report.watchlist_hits)


def test_main_json_reports_violation_and_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = tmp_path / "fake.md"
    doc.write_text("See `polylogue/nonexistent/module.py` for the implementation.\n")
    monkeypatch.setattr(verify_docs_drift, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(verify_docs_drift, "REFERENCE_DOCS", ("fake.md",))

    assert verify_docs_drift.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["missing_paths"][0]["quoted"] == "polylogue/nonexistent/module.py"


def test_main_reports_ok_with_no_reference_docs(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_docs_drift, "REFERENCE_DOCS", ())
    assert verify_docs_drift.main([]) == 0
    assert "zero unhandled drift" in capsys.readouterr().out
