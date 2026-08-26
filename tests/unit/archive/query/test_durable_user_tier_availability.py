"""Fail-closed assertion reads when the durable user tier is unavailable."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.archive.query.predicate import QueryPredicate
from polylogue.archive.query.transaction import run_archive_read, run_archive_read_sync
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def _assertion_predicate() -> QueryPredicate:
    source = parse_unit_source_expression("assertions where kind:note")
    assert source is not None
    return source.predicate


def _archive_without_user_tier(root: Path) -> Path:
    with ArchiveStore(root):
        pass
    (root / "user.db").unlink()
    return root


def test_assertion_rows_and_counts_fail_before_sql_when_user_tier_is_missing(tmp_path: Path) -> None:
    root = _archive_without_user_tier(tmp_path / "archive")
    predicate = _assertion_predicate()

    with pytest.raises(ArchiveTierUnavailableError) as rows_error:
        run_archive_read_sync(
            root,
            operation="assertion-rows",
            arguments={},
            work=lambda archive: archive.query_assertions(predicate),
        )
    with pytest.raises(ArchiveTierUnavailableError) as counts_error:
        run_archive_read_sync(
            root,
            operation="assertion-counts",
            arguments={},
            work=lambda archive: archive.query_unit_counts("assertion", predicate),
        )

    for error in (rows_error.value, counts_error.value):
        assert error.code == "archive_tier_unavailable"
        assert error.path == str((root / "user.db").resolve())
        assert "restore or initialize" in str(error)
    assert not (root / "user.db").exists()


@pytest.mark.asyncio
async def test_async_assertion_multi_counts_uses_the_same_durable_precondition(tmp_path: Path) -> None:
    root = _archive_without_user_tier(tmp_path / "archive")
    predicate = _assertion_predicate()

    with pytest.raises(ArchiveTierUnavailableError):
        await run_archive_read(
            root,
            operation="assertion-multi-counts",
            arguments={},
            work=lambda archive: archive.query_unit_multi_counts("assertion", predicate, group_by=("status", "kind")),
        )


def test_initialized_empty_user_tier_is_a_real_empty_result(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    with ArchiveStore(root):
        pass

    predicate = _assertion_predicate()
    with ArchiveStore.open_existing(root) as archive:
        assert archive.query_assertions(predicate) == []
        assert archive.query_unit_counts("assertion", predicate) == []


@pytest.mark.parametrize("replacement", ["directory", "corrupt"])
def test_invalid_user_tier_is_not_confused_with_an_empty_tier(tmp_path: Path, replacement: str) -> None:
    root = _archive_without_user_tier(tmp_path / "archive")
    user_db = root / "user.db"
    if replacement == "directory":
        user_db.mkdir()
    else:
        user_db.write_bytes(b"not sqlite")

    with pytest.raises(ArchiveTierUnavailableError) as error:
        run_archive_read_sync(
            root,
            operation="assertion-rows",
            arguments={},
            work=lambda archive: archive.query_assertions(_assertion_predicate()),
        )

    assert error.value.path == str(user_db.resolve())
    assert error.value.reason
