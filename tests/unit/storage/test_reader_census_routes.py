"""Migrated archive readers hold a write-denying connection on their real route.

Each test drives a production route against a bootstrapped archive while a
spy on the module's own ``open_readonly_connection`` attempts the full
mutation matrix on the very connection the route receives, then lets the
route read through it. Anti-vacuity: restoring the former raw
``sqlite3.connect(f"file:...?mode=ro")`` bypasses the spy (the route-level
``opened`` assertion fails), and such a connection also accepts writable
PRAGMAs and a writable ATTACH, which the matrix rejects.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from polylogue.analysis import delegation_work_evidence_materializer as delegation
from polylogue.browser_capture import receiver
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.live import convergence_debt_retry, hook_tool_response, production_baseline
from polylogue.storage import usage
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.archive_templates import bootstrap_archive_root

# Denied at SQLite's boundary (authorizer or query_only), never by a constraint.
DENIED = "not authorized|readonly database"


def assert_write_denied(conn: sqlite3.Connection, scratch: Path) -> None:
    table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1").fetchone()[0]
    writable = scratch / f"attached-{id(conn)}.db"
    for statement in (
        f'INSERT INTO "{table}" DEFAULT VALUES',
        f'UPDATE "{table}" SET rowid = rowid',
        f'DELETE FROM "{table}"',
        "CREATE TABLE forbidden (value INTEGER)",
        "PRAGMA query_only = OFF",
        "PRAGMA user_version = 7",
    ):
        with pytest.raises(sqlite3.Error, match=DENIED):
            conn.execute(statement)
    with pytest.raises(sqlite3.Error, match=DENIED):
        conn.execute("ATTACH DATABASE ? AS writable", (str(writable),))
    assert not writable.exists()
    assert conn.execute("PRAGMA query_only").fetchone()[0] == 1


def spy(monkeypatch: pytest.MonkeyPatch, module: ModuleType, scratch: Path) -> list[Path]:
    opened: list[Path] = []

    def audited(path: str | Path, **kwargs: Any) -> sqlite3.Connection:
        conn = open_readonly_connection(path, **kwargs)
        assert_write_denied(conn, scratch)
        opened.append(Path(path).resolve())
        return conn

    monkeypatch.setattr(module, "open_readonly_connection", audited)
    return opened


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    return bootstrap_archive_root(tmp_path / "archive")


def test_usage_report_reads_user_and_index_through_profiles(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = spy(monkeypatch, usage, tmp_path)
    report = usage.origin_usage_report_for_archive_root(archive)
    assert report.origins == ()
    assert opened == [(archive / "user.db").resolve(), (archive / "index.db").resolve()]


def test_delegation_freshness_probe_reads_index_through_profile(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe reads through the pinned operation-read boundary (#5722).

    Anti-vacuity: a probe that opens its own writable connection bypasses the
    spy, so ``opened`` stays empty; a pinned archive connection without the
    read authorizer accepts the mutation matrix.
    """
    opened: list[Path] = []

    @contextmanager
    def audited(root: Path, **kwargs: Any) -> Iterator[Any]:
        with open_operation_read(root, **kwargs) as pinned:
            assert_write_denied(pinned.archive._conn, tmp_path)
            opened.append(Path(pinned.archive.index_db_path).resolve())
            yield pinned

    monkeypatch.setattr(delegation, "open_operation_read", audited)
    assert delegation.delegation_work_evidence_materialization_needed(archive) is True
    assert opened == [(archive / "index.db").resolve()]


def test_browser_receiver_lookups_read_through_profiles(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = spy(monkeypatch, receiver, tmp_path)
    raw = receiver._lookup_raw_archive_state(
        archive, provider="chatgpt", provider_session_id="absent", artifact_ref="artifact"
    )
    index = receiver._lookup_index_archive_state(archive, raw_id=None, provider="chatgpt", provider_session_id="absent")
    assert raw == receiver._RawArchiveLookup() and index == receiver._IndexArchiveLookup()
    assert [path.name for path in opened] == ["source.db", "index.db"]


def test_convergence_debt_lookup_attaches_source_read_only(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = spy(monkeypatch, convergence_debt_retry, tmp_path)
    assert convergence_debt_retry._archive_convergence_debt_source_path_from_root(archive, "absent") is None
    assert [path.name for path in opened] == ["index.db"]


def test_retained_hook_read_has_no_archive_root_reopen_and_baseline_uses_profile(
    archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not hasattr(hook_tool_response, "resolve_hook_tool_responses")
    baseline = spy(monkeypatch, production_baseline, tmp_path)
    decisions = production_baseline.unretained_source_decisions(
        SimpleNamespace(accepted=()),  # type: ignore[arg-type]
        archive / "source.db",
    )
    assert decisions == ()
    assert [path.name for path in baseline] == ["source.db"]


def test_async_backend_readers_deny_writes_including_attached_siblings(tmp_path: Path) -> None:
    """The aiosqlite read pool attached durable siblings writable and had no authorizer."""
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

    archive = bootstrap_archive_root(tmp_path / "archive")
    backend = SQLiteBackend(db_path=archive / "index.db")
    writable = tmp_path / "writable.db"

    async def probe() -> None:
        async with backend._get_read_connection() as conn:
            cursor = await conn.execute("PRAGMA database_list")
            attached = {str(row[1]) for row in await cursor.fetchall()}
            assert "user_tier" in attached
            cursor = await conn.execute("SELECT count(*) FROM sessions")
            row = await cursor.fetchone()
            assert row is not None and row[0] == 0
            for statement in (
                "DELETE FROM sessions",
                "DELETE FROM user_tier.assertions",
                "CREATE TABLE forbidden (value INTEGER)",
                "PRAGMA query_only = OFF",
                "PRAGMA user_tier.user_version = 7",
            ):
                with pytest.raises(sqlite3.Error, match=DENIED):
                    await conn.execute(statement)
            with pytest.raises(sqlite3.Error, match=DENIED):
                await conn.execute("ATTACH DATABASE ? AS writable", (str(writable),))
        await backend.close()

    asyncio.run(probe())
    assert not writable.exists()
    with closing(sqlite3.connect(archive / "user.db")) as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'forbidden'").fetchone()[0] == 0
