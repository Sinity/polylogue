from __future__ import annotations

from pathlib import Path

from devtools.verify_archive_resolver_completeness import (
    BASELINE_CALL_SITES,
    ROOT,
    ResolverCallSite,
    find_call_sites,
    main,
    unbaselined_call_sites,
)


def test_live_repo_has_no_unbaselined_call_sites() -> None:
    """The real repo's current call sites must all already be in the recorded baseline."""
    assert main(["--json"]) == 0


def test_find_call_sites_detects_a_seeded_call(tmp_path: Path) -> None:
    """A seeded call to one of the four resolvers must be detected."""
    module = tmp_path / "fake_reader.py"
    module.write_text("db_path = active_index_db_path()\n", encoding="utf-8")

    hits = find_call_sites(roots=(tmp_path,))

    assert len(hits) == 1
    assert hits[0].function == "active_index_db_path"
    assert hits[0].path == module


def test_unbaselined_call_sites_flags_a_file_outside_the_recorded_baseline(tmp_path: Path) -> None:
    """A call site in a file NOT already recorded in BASELINE_CALL_SITES must be flagged."""
    module = tmp_path / "fake_new_reader.py"
    module.write_text("db_path = sibling_index_db(anchor)\n", encoding="utf-8")

    hits = find_call_sites(roots=(tmp_path,))
    unbaselined = unbaselined_call_sites(hits, repo_root=tmp_path)

    assert len(unbaselined) == 1
    assert unbaselined[0].function == "sibling_index_db"
    assert unbaselined[0].path == module


def test_unbaselined_call_sites_accepts_a_baselined_file() -> None:
    """A call site in a file already recorded in the baseline must not be flagged."""
    baselined_path, *_ = sorted(BASELINE_CALL_SITES["active_index_db_path"])
    hit = ResolverCallSite(
        function="active_index_db_path",
        path=ROOT / baselined_path,
        lineno=1,
        line="db_path = active_index_db_path()",
    )

    unbaselined = unbaselined_call_sites([hit])

    assert unbaselined == []
