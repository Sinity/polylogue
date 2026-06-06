from __future__ import annotations

from devtools.verify_test_coverage_contracts import ROOT, _test_for_module


def test_archive_tiers_modules_match_flat_storage_test_files() -> None:
    assert _test_for_module(ROOT / "polylogue/storage/sqlite/archive_tiers/write.py") == (
        ROOT / "tests/unit/storage/test_archive_tiers_write.py"
    )
    assert _test_for_module(ROOT / "polylogue/storage/sqlite/archive_tiers/source_write.py") == (
        ROOT / "tests/unit/storage/test_archive_tiers_source_write.py"
    )
    assert _test_for_module(ROOT / "polylogue/storage/sqlite/archive_tiers/user_write.py") == (
        ROOT / "tests/unit/storage/test_archive_tiers_user_write.py"
    )
    assert _test_for_module(ROOT / "polylogue/storage/sqlite/archive_tiers/ops_write.py") == (
        ROOT / "tests/unit/storage/test_archive_tiers_ops_write.py"
    )


def test_parser_modules_match_flat_parser_test_files() -> None:
    assert _test_for_module(ROOT / "polylogue/sources/parsers/codex.py") == (
        ROOT / "tests/unit/sources/test_parsers_codex.py"
    )
