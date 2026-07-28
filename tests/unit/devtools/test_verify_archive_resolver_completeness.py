from __future__ import annotations

from pathlib import Path

from devtools.verify_archive_resolver_completeness import (
    ROOT,
    ResolverCallSite,
    find_call_sites,
    main,
    unbaselined_call_sites,
)

_FAKE_RESOLVER = "fake_duplicate_resolver"


def test_live_repo_has_no_unbaselined_call_sites() -> None:
    """The real repo's current call sites must all already be in the recorded baseline.

    All four originally named resolvers (``active_index_db_path``,
    ``resolve_active_index_db_path``, ``sibling_index_db``,
    ``archive_file_set_root_for_paths``) are fully migrated and removed as of
    polylogue-l2cd, so ``BASELINE_CALL_SITES`` is empty and this should
    trivially pass with zero call sites found.
    """
    assert main(["--json"]) == 0


def test_find_call_sites_detects_a_seeded_call(tmp_path: Path) -> None:
    """A seeded call to a tracked resolver name must be detected."""
    module = tmp_path / "fake_reader.py"
    module.write_text(f"db_path = {_FAKE_RESOLVER}()\n", encoding="utf-8")

    hits = find_call_sites(roots=(tmp_path,), functions=(_FAKE_RESOLVER,))

    assert len(hits) == 1
    assert hits[0].function == _FAKE_RESOLVER
    assert hits[0].path == module


def test_unbaselined_call_sites_flags_a_file_outside_the_recorded_baseline(tmp_path: Path) -> None:
    """A call site in a file NOT already recorded in the baseline must be flagged."""
    module = tmp_path / "fake_new_reader.py"
    module.write_text(f"db_path = {_FAKE_RESOLVER}()\n", encoding="utf-8")

    hits = find_call_sites(roots=(tmp_path,), functions=(_FAKE_RESOLVER,))
    unbaselined = unbaselined_call_sites(hits, baseline={}, repo_root=tmp_path)

    assert len(unbaselined) == 1
    assert unbaselined[0].function == _FAKE_RESOLVER
    assert unbaselined[0].path == module


def test_unbaselined_call_sites_accepts_a_baselined_file() -> None:
    """A call site in a file already recorded in the baseline must not be flagged."""
    baselined_path = "polylogue/paths/_roots.py"
    hit = ResolverCallSite(
        function=_FAKE_RESOLVER,
        path=ROOT / baselined_path,
        lineno=1,
        line=f"db_path = {_FAKE_RESOLVER}()",
    )

    unbaselined = unbaselined_call_sites(
        [hit],
        baseline={_FAKE_RESOLVER: frozenset({baselined_path})},
    )

    assert unbaselined == []
