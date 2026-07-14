from __future__ import annotations

from devtools import verify_docs_coverage


def test_collect_coverage_passes_on_current_tree() -> None:
    """The lint must be green on HEAD -- a live regression here means either a
    real undocumented surface shipped, or the baseline drifted out of sync."""
    report = verify_docs_coverage.collect_coverage()
    assert report.ok, [f"{g.surface}: {g.name}" for g in report.gaps]


def test_collect_coverage_flags_an_undocumented_cli_command(monkeypatch) -> None:
    """Anti-vacuity: a CLI command absent from both the docs tree and the
    baseline must fail the lane, naming the exact missing command."""
    monkeypatch.setattr(verify_docs_coverage, "_cli_inventory", lambda: ("totally-unfamiliar-verb",))
    monkeypatch.setattr(verify_docs_coverage, "_mcp_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_config_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_route_inventory", lambda: ())

    report = verify_docs_coverage.collect_coverage(docs_text="nothing relevant here")

    assert not report.ok
    assert report.gaps == (verify_docs_coverage.CoverageGap(surface="cli", name="totally-unfamiliar-verb"),)


def test_collect_coverage_baseline_entry_suppresses_a_gap(monkeypatch) -> None:
    """A surface explicitly recorded in the baseline is tracked debt, not a
    failure -- this is what lets the gate ship without a full doc-writing pass
    for the existing backlog."""
    monkeypatch.setattr(verify_docs_coverage, "_cli_inventory", lambda: ("baselined-verb",))
    monkeypatch.setattr(verify_docs_coverage, "_mcp_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_config_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_route_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_load_baseline", lambda: {"cli": {"baselined-verb": "known debt"}})

    report = verify_docs_coverage.collect_coverage(docs_text="nothing relevant here")

    assert report.ok


def test_collect_coverage_reports_stale_baseline_entries(monkeypatch) -> None:
    """A baseline entry whose surface is now documented is reported as stale
    (should be removed) but does not fail the lane -- keeps the baseline from
    silently outliving the debt it records."""
    monkeypatch.setattr(verify_docs_coverage, "_cli_inventory", lambda: ("now-documented-verb",))
    monkeypatch.setattr(verify_docs_coverage, "_mcp_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_config_inventory", lambda: ())
    monkeypatch.setattr(verify_docs_coverage, "_route_inventory", lambda: ())
    monkeypatch.setattr(
        verify_docs_coverage, "_load_baseline", lambda: {"cli": {"now-documented-verb": "stale reason"}}
    )

    report = verify_docs_coverage.collect_coverage(docs_text="the now-documented-verb command does X")

    assert report.ok
    assert report.stale_baseline == (verify_docs_coverage.CoverageGap(surface="cli", name="now-documented-verb"),)


def test_baseline_file_parses_and_matches_live_gaps() -> None:
    """The committed baseline file must be well-formed YAML honoring the
    documented schema, and every entry in it must correspond to a real,
    currently-undocumented surface -- catches a stale or hand-typo'd entry."""
    baseline = verify_docs_coverage._load_baseline()
    assert baseline, "expected at least one tracked surface class in the baseline"
    for surface, entries in baseline.items():
        assert surface in {"cli", "mcp", "config", "route"}
        for name, reason in entries.items():
            assert isinstance(name, str) and name
            assert isinstance(reason, str) and reason
