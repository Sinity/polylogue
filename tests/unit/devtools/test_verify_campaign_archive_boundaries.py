from __future__ import annotations

from pathlib import Path

from devtools.verify_campaign_archive_boundaries import (
    _missing_ownership_references,
    _scan_modules,
    main,
)


def test_live_campaign_modules_are_clean() -> None:
    """The real devtools campaign modules must currently pass this lint."""
    assert main(["--json"]) == 0


def test_scan_modules_flags_seeded_benchmark_db_sentinel(tmp_path: Path) -> None:
    """A seeded ambiguous 'benchmark.db' sentinel must be caught."""
    module = tmp_path / "fake_campaign.py"
    module.write_text(
        'db_path = archive_dir / "benchmark.db"\nrun_synthetic_benchmark_campaign(name, db_path)\n',
        encoding="utf-8",
    )

    sentinel_hits, sibling_hits = _scan_modules((module,))

    assert len(sentinel_hits) == 1
    assert sentinel_hits[0].path == module
    assert "benchmark.db" in sentinel_hits[0].line
    # The seeded fixture also derives a sibling path directly.
    assert sibling_hits == []


def test_scan_modules_flags_seeded_sibling_tier_derivation(tmp_path: Path) -> None:
    """A seeded ad hoc sibling tier-path derivation (bypassing ArchiveLocation) must be caught."""
    module = tmp_path / "fake_campaign_root.py"
    module.write_text(
        'db_path = output_dir / "index.db"\nbackend = SQLiteBackend(db_path=db_path)\n',
        encoding="utf-8",
    )

    sentinel_hits, sibling_hits = _scan_modules((module,))

    assert sentinel_hits == []
    assert len(sibling_hits) == 1
    assert sibling_hits[0].tier_filename == "index.db"
    assert sibling_hits[0].path == module


def test_scan_modules_ignores_resolver_module_by_name(tmp_path: Path) -> None:
    """The sanctioned resolver module itself is exempt (it legitimately names tier files once)."""
    module = tmp_path / "archive_identity.py"
    module.write_text('active_index_path = root / "index.db"\n', encoding="utf-8")

    sentinel_hits, sibling_hits = _scan_modules((module,))

    assert sentinel_hits == []
    assert sibling_hits == []


def test_missing_ownership_reference_flags_seeded_entry_point_without_campaign_archive_location(
    tmp_path: Path,
) -> None:
    """An entry point missing CampaignArchiveLocation plumbing must be caught."""
    module = tmp_path / "fake_run_campaign.py"
    module.write_text(
        "async def _run(args):\n    return await generate_archive(spec, archive_dir)\n",
        encoding="utf-8",
    )

    missing = _missing_ownership_references(((module, "_run"),))

    assert len(missing) == 1
    assert missing[0].path == module
    assert missing[0].symbol == "_run"


def test_missing_ownership_reference_passes_when_symbol_references_campaign_archive_location(
    tmp_path: Path,
) -> None:
    module = tmp_path / "fake_run_campaign_ok.py"
    module.write_text(
        "async def _run(args):\n"
        "    with CampaignArchiveLocation.acquire(archive_dir) as location:\n"
        "        return await generate_archive(spec, archive_dir, location=location)\n",
        encoding="utf-8",
    )

    missing = _missing_ownership_references(((module, "_run"),))

    assert missing == []
