"""Manual continuation validates cycles and stores replayable User authority."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.operations.facade_writers import record_manual_continuation_product
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.write import ManualContinuationAuthorityError
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.index_writer import write_fixture_index_session


def _parsed(native: str, body: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native,
        messages=[ParsedMessage(provider_message_id="m1", role=Role.USER, text=body)],
    )


def _archive_with_sessions(tmp_path: Path) -> Config:
    with ArchiveStore(tmp_path):
        pass
    conn = connect_measured(tmp_path / "index.db")
    try:
        for native_id in ("child", "parent"):
            write_fixture_index_session(conn, _parsed(native_id, "original"))
    finally:
        conn.close()
    # The facade forwards this mutation to the resident daemon; the daemon
    # runs the product writer exercised here.
    return Config(archive_root=tmp_path, render_root=tmp_path / "render", sources=[], db_path=tmp_path / "index.db")


def _record_manual(config: Config, child: str, parent: str) -> None:
    asyncio.run(
        run_archive_fixture_write(
            config.archive_root, lambda: record_manual_continuation_product(config, child, parent)
        )
    )


def test_manual_continuation_records_a_storable_lineage_edge(tmp_path: Path) -> None:
    """The edge lands, marked spawned-fresh and resolved to the named parent.

    Anti-vacuity: restore ``status='resolved'`` in the INSERT and this fails
    with ``IntegrityError: CHECK constraint failed``; restore
    ``author_ref="service:polylogue"`` and it fails with ``unsupported object
    ref kind``; drop ``resolved_dst_session_id`` instead and the resolved
    parent assertion goes red, so the test is not merely proving the write
    did not raise. The handoff assertion is read back so the second fix is
    covered by an outcome, not by the absence of an exception.
    """
    archive = _archive_with_sessions(tmp_path)

    _record_manual(archive, "codex-session:child", "codex-session:parent")

    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        rows = conn.execute(
            "SELECT src_session_id, link_type, inheritance, status, resolved_dst_session_id, "
            "branch_point_message_id, method FROM session_links"
        ).fetchall()
    finally:
        conn.close()

    assert rows == [
        (
            "codex-session:child",
            "continuation",
            "spawned-fresh",
            None,
            "codex-session:parent",
            None,
            "manual-continuation",
        )
    ]

    user = sqlite3.connect(tmp_path / "user.db")
    try:
        value = json.loads(user.execute("SELECT value_json FROM assertions").fetchone()[0])
        handoffs = user.execute("SELECT target_ref, kind, author_ref, author_kind, status FROM assertions").fetchall()
    finally:
        user.close()
    assert value == {"_schema": "polylogue.manual-continuation.v1", "parent_session_id": "codex-session:parent"}
    assert handoffs == [("session:codex-session:child", "handoff", "actor:polylogue", "service", "candidate")]


def test_manual_continuation_still_refuses_an_absent_parent(tmp_path: Path) -> None:
    """The route's own preconditions are unchanged by the storage fix."""
    archive = _archive_with_sessions(tmp_path)

    with pytest.raises(ValueError, match="parent session does not exist"):
        _record_manual(archive, "codex-session:child", "codex-session:absent")


def test_manual_continuation_refuses_an_edge_that_closes_a_cycle(tmp_path: Path) -> None:
    """With child already under parent, recording parent as child's continuation is refused.

    Anti-vacuity: write the resolved edge without the explicit cycle check and
    the resolver never examines it, so both sessions publish each other as
    ``parent_session_id``.
    """
    archive = _archive_with_sessions(tmp_path)
    _record_manual(archive, "codex-session:child", "codex-session:parent")

    with pytest.raises(ValueError, match="manual continuation refused"):
        _record_manual(archive, "codex-session:parent", "codex-session:child")

    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        links = conn.execute("SELECT src_session_id, resolved_dst_session_id FROM session_links").fetchall()
        parents = dict(conn.execute("SELECT session_id, parent_session_id FROM sessions").fetchall())
    finally:
        conn.close()
    assert links == [("codex-session:child", "codex-session:parent")]
    assert parents["codex-session:parent"] is None


@pytest.mark.parametrize("existing_hook", [False, True])
def test_manual_authority_rederives_after_canonical_full_replacement(tmp_path: Path, existing_hook: bool) -> None:
    config = _archive_with_sessions(tmp_path)
    if existing_hook:
        with sqlite3.connect(tmp_path / "index.db") as index:
            index.execute(
                "INSERT INTO session_links(src_session_id,dst_origin,dst_native_id,link_type,inheritance,method,confidence,observed_at_ms) "
                "VALUES ('codex-session:child','codex-session','parent','continuation','prefix-sharing',"
                "'authoritative-hook-evidence',1.0,1)"
            )
    _record_manual(config, "codex-session:child", "codex-session:parent")
    with connect_measured(tmp_path / "index.db") as index:
        write_fixture_index_session(index, _parsed("child", "replacement"), force_replace=True)
        edge = tuple(
            index.execute(
                "SELECT method,inheritance,resolved_dst_session_id FROM session_links WHERE src_session_id='codex-session:child'"
            ).fetchone()
        )
        assert edge == ("manual-continuation", "spawned-fresh", "codex-session:parent")
        assert (
            index.execute("SELECT text FROM blocks WHERE session_id='codex-session:child'").fetchone()[0]
            == "replacement"
        )
        assert (
            index.execute("SELECT parent_session_id FROM sessions WHERE native_id='child'").fetchone()[0]
            == "codex-session:parent"
        )


@pytest.mark.parametrize("authority", ["prose-only", "deleted", "malformed"])
def test_canonical_replacement_uses_only_explicit_current_manual_authority(tmp_path: Path, authority: str) -> None:
    config = _archive_with_sessions(tmp_path)
    _record_manual(config, "codex-session:child", "codex-session:parent")
    with sqlite3.connect(tmp_path / "user.db") as user:
        if authority == "deleted":
            user.execute("UPDATE assertions SET status='deleted'")
        elif authority == "prose-only":
            user.execute("UPDATE assertions SET value_json=NULL")
        else:
            user.execute(
                "UPDATE assertions SET value_json=?", (json.dumps({"_schema": "polylogue.manual-continuation.v1"}),)
            )
    with connect_measured(tmp_path / "index.db") as index:
        if authority == "malformed":
            with pytest.raises(ManualContinuationAuthorityError):
                write_fixture_index_session(index, _parsed("child", "replacement"), force_replace=True)
            assert (
                index.execute("SELECT text FROM blocks WHERE session_id='codex-session:child'").fetchone()[0]
                == "original"
            )
        else:
            write_fixture_index_session(index, _parsed("child", "replacement"), force_replace=True)
            assert (
                index.execute(
                    "SELECT COUNT(*) FROM session_links WHERE src_session_id='codex-session:child'"
                ).fetchone()[0]
                == 0
            )
            assert index.execute("SELECT parent_session_id FROM sessions WHERE native_id='child'").fetchone()[0] is None


def test_failed_manual_projection_retains_replayable_user_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.storage.sqlite.archive_tiers import write

    config = _archive_with_sessions(tmp_path)
    original_project = write._project_manual_continuations

    def fail_projection(*args: object) -> None:
        raise RuntimeError("neutral projection interruption")

    monkeypatch.setattr(write, "_project_manual_continuations", fail_projection)
    with pytest.raises(RuntimeError, match="neutral projection interruption"):
        _record_manual(config, "codex-session:child", "codex-session:parent")
    with sqlite3.connect(tmp_path / "user.db") as user:
        assert (
            json.loads(user.execute("SELECT value_json FROM assertions").fetchone()[0])["parent_session_id"]
            == "codex-session:parent"
        )
    monkeypatch.setattr(write, "_project_manual_continuations", original_project)
    with connect_measured(tmp_path / "index.db") as index:
        write_fixture_index_session(index, _parsed("child", "replacement"), force_replace=True)
        assert (
            index.execute("SELECT parent_session_id FROM sessions WHERE native_id='child'").fetchone()[0]
            == "codex-session:parent"
        )


def test_manual_missing_user_authority_refuses_without_recreating_tier(tmp_path: Path) -> None:
    from polylogue.core.errors import ArchiveTierUnavailableError

    config = _archive_with_sessions(tmp_path)
    (tmp_path / "user.db").rename(tmp_path / "retained-user.db")
    with pytest.raises(ArchiveTierUnavailableError) as failure:
        _record_manual(config, "codex-session:child", "codex-session:parent")
    assert failure.value.tier == "user.db"
    assert not (tmp_path / "user.db").exists()
    with sqlite3.connect(tmp_path / "index.db") as index:
        assert index.execute("SELECT COUNT(*) FROM session_links").fetchone()[0] == 0


def test_append_rederives_deleted_manual_authority_without_preserving_index_edge(tmp_path: Path) -> None:
    config = _archive_with_sessions(tmp_path)
    _record_manual(config, "codex-session:child", "codex-session:parent")
    with sqlite3.connect(tmp_path / "user.db") as user:
        user.execute("UPDATE assertions SET status='deleted'")
    delta = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="child",
        messages=[ParsedMessage(provider_message_id="m2", role=Role.USER, text="appended")],
    )
    with connect_measured(tmp_path / "index.db") as index:
        write_fixture_index_session(index, delta, merge_append=True)
        assert (
            index.execute("SELECT COUNT(*) FROM session_links WHERE src_session_id='codex-session:child'").fetchone()[0]
            == 0
        )
        assert index.execute("SELECT parent_session_id FROM sessions WHERE native_id='child'").fetchone()[0] is None
        assert [
            row[0]
            for row in index.execute("SELECT text FROM blocks WHERE session_id='codex-session:child' ORDER BY block_id")
        ] == ["original", "appended"]
