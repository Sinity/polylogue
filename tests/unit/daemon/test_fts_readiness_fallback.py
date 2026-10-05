"""FTS readiness reports the current authoritative input/output relation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.fts import completed_fts_readiness
from tests.infra.index_writer import write_fixture_index_session


def test_exact_coverage_counts_tool_blocks_as_indexable(tmp_path: Path) -> None:
    """Exact coverage must use the FTS-population predicate (search_text != '').

    A tool_use block carries a derived search_text but a NULL display text. The
    FTS index includes it, so the exact source count must include it too —
    otherwise indexed/source exceeds 100%.
    """
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = connect_measured(db)
    # The archive writer reads rows by column name, as on production connections.
    conn.row_factory = sqlite3.Row
    try:
        session = ParsedSession(
            source_name=Provider.CODEX,
            provider_session_id="cov-tool-1",
            title="tool coverage",
            messages=[
                ParsedMessage(
                    provider_message_id="u1",
                    role=Role.USER,
                    text="run the build",
                    position=0,
                    blocks=[ParsedContentBlock(type=BlockType.TEXT, text="run the build")],
                ),
                ParsedMessage(
                    provider_message_id="a1",
                    role=Role.ASSISTANT,
                    text=None,
                    position=1,
                    blocks=[
                        ParsedContentBlock(
                            type=BlockType.TOOL_USE,
                            tool_name="exec_command",
                            tool_id="t1",
                            tool_input={"command": "make build"},
                        ),
                    ],
                ),
            ],
        )
        write_fixture_index_session(conn, session)
        conn.commit()
        text_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE text IS NOT NULL").fetchone()[0])
        search_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0])
    finally:
        conn.close()

    # The tool block makes the two predicates diverge; the exact coverage must
    # use search_text (the FTS predicate), not text.
    assert search_blocks > text_blocks

    fts = completed_fts_readiness(db, lambda: fts_readiness_info(db, exact=True))
    assert fts["coverage_pct"] == 100.0
    assert fts["messages_ready"] is True


def test_genuinely_empty_archive_reports_coverage_as_unmeasured_not_exact(tmp_path: Path) -> None:
    """polylogue-oitx: a freshly initialized, genuinely empty archive has a
    zero-denominator coverage_pct (0 indexable rows). ``invariant_ready``
    only proves triggers/tables exist -- it is not evidence of measured
    coverage, so this must report ``None`` (unmeasured), never a fabricated
    ``100.0``/``0.0`` derived from ``invariant_ready`` alone.
    """
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)

    fts = completed_fts_readiness(db, lambda: fts_readiness_info(db, exact=False))

    assert fts["message_indexable_count"] == 0
    assert fts["message_indexed_count"] == 0
    assert fts["coverage_pct"] is None


def test_genuinely_empty_archive_reports_coverage_as_unmeasured_exact(tmp_path: Path) -> None:
    """Same as above, through the exact (invariant-snapshot) path."""
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)

    fts = completed_fts_readiness(db, lambda: fts_readiness_info(db, exact=True))

    assert fts["message_indexable_count"] == 0
    assert fts["message_indexed_count"] == 0
    assert fts["coverage_pct"] is None


def test_unreadable_archive_index_reports_nothing_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-bu47u: a failed readiness query certifies nothing.

    A readiness flag used to return ``True`` from inside the
    ``except sqlite3.Error`` handler while every sibling key returned the
    not-ready value -- a positive readiness claim emitted by an error path.

    Anti-vacuity: restore ``"messages_ready": True`` (or the fabricated
    ``coverage_pct: 0.0``) in that handler and this fails.
    """
    from polylogue.daemon import fts_status

    index = tmp_path / "index.db"
    initialize_archive_database(index, ArchiveTier.INDEX)

    def explode(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.Error("simulated readiness query failure")

    monkeypatch.setattr(fts_status, "open_readonly_connection", explode)

    payload = fts_status._archive_readiness_info(index, exact=False)

    assert payload is not None
    assert payload["messages_ready"] is False
    assert payload["invariant_ready"] is False
    assert payload["coverage_pct"] is None
    assert payload["coverage_exact"] is False


def test_unreadable_index_publishes_unknown_coverage_not_a_measured_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that could not open the index never reports 0.0% coverage.

    ``fts_readiness_info``'s own ``except sqlite3.Error`` handler returned
    ``coverage_pct: 0.0`` -- a measured empty index for a database nothing
    opened. The daemon status route happened to mask it through
    ``FTSReadiness``' unreadable-coverage validator, but ``status_snapshot``'s
    minimal path publishes this dict directly, so ``/api/status`` and the
    plaintext CLI reported "0.0% indexed" for an unreadable archive
    (polylogue-20d.17.4 AC3).

    Anti-vacuity: restore ``"coverage_pct": 0.0`` in that handler and both the
    producer assertion and the ``/api/status`` assertion below go red.
    """
    from unittest.mock import patch

    from polylogue.daemon import fts_status, status_snapshot

    index = tmp_path / "index.db"
    index.write_bytes(b"not a sqlite database at all")

    def explode(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(fts_status, "open_readonly_connection", explode)

    payload = fts_readiness_info(index)

    assert payload["coverage_pct"] is None
    assert payload["coverage_exact"] is False
    assert payload["message_indexed_count"] is None
    assert payload["message_indexable_count"] is None
    assert payload["messages_ready"] is False
    assert payload["invariant_ready"] is False
    assert "file is not a database" in str(payload["unavailable_reason"])

    # The same value, unaltered, on the surface that publishes it directly.
    with patch.object(status_snapshot, "resolve_active_index_path", lambda *_a, **_k: index):
        status_payload = status_snapshot._minimal_status_payload()
    fts_readiness = status_payload["fts_readiness"]
    assert isinstance(fts_readiness, dict)
    assert fts_readiness["coverage_pct"] is None


def test_a_genuinely_empty_index_still_reports_its_measured_coverage(tmp_path: Path) -> None:
    """The control: an index that *was* read keeps its measured answer.

    Without this, the test above could pass from a blanket "coverage is always
    unknown" change.
    """
    index = tmp_path / "index.db"
    initialize_archive_database(index, ArchiveTier.INDEX)

    payload = fts_readiness_info(index, exact=True)

    assert payload["message_indexable_count"] == 0
    assert payload["message_indexed_count"] == 0
    assert "unavailable_reason" not in payload


def _text_session(native_id: str, text: str) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=native_id,
        messages=[
            ParsedMessage(
                provider_message_id=f"{native_id}-u1",
                role=Role.USER,
                text=text,
                position=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
            )
        ],
    )


def test_unbound_archive_fallback_measures_every_count_in_one_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status fallback reads source and indexed counts from one snapshot.

    With no standing readiness binding, ``_archive_readiness_info`` runs the
    global inspection as separate COUNT statements. A daemon commit of new
    blocks (and their trigger-maintained FTS rows) lands right after the
    block-source count. Both committed states are consistent, so the report
    must be too: indexed rows equal source rows and coverage is 100%.

    Anti-vacuity: drop the ``BEGIN`` in ``_archive_readiness_info`` and the
    later docsize count sees the new rows while the source count does not,
    so ``indexed_rows`` exceeds ``source_rows`` and coverage exceeds 100%.
    """
    from polylogue.daemon import fts_status
    from polylogue.storage.fts.derivation import fts_readiness_binding
    from tests.infra.snapshot_probe import CommitBetweenStatements

    index = tmp_path / "index.db"
    initialize_archive_database(index, ArchiveTier.INDEX)
    writer = connect_measured(index)
    try:
        assert str(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() == "wal"
        write_fixture_index_session(writer, _text_session("snapshot-first", "first committed text"))
        writer.commit()
        assert fts_readiness_binding(writer) is None
        committed_before = int(writer.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0])
        assert committed_before > 0

        def concurrent_commit() -> None:
            write_fixture_index_session(writer, _text_session("snapshot-second", "second committed text"))
            writer.commit()

        from polylogue.storage.sqlite.connection_profile import open_readonly_connection as real_open

        probes: list[CommitBetweenStatements] = []

        def open_probe(*args: object, **kwargs: object) -> CommitBetweenStatements:
            probe = CommitBetweenStatements(
                real_open(*args, **kwargs),  # type: ignore[arg-type]
                trigger_sql="SELECT COUNT(*) FROM blocks WHERE search_text != ''",
                commit=concurrent_commit,
            )
            probes.append(probe)
            return probe

        monkeypatch.setattr(fts_status, "open_readonly_connection", open_probe)
        payload = fts_status._archive_readiness_info(index, exact=False)
        committed_after = int(writer.execute("SELECT COUNT(*) FROM blocks WHERE search_text != ''").fetchone()[0])
    finally:
        writer.close()

    assert [probe.fired for probe in probes] == [True]
    assert committed_after > committed_before
    assert payload is not None
    surface = payload["surfaces"]["messages_fts"]  # type: ignore[index]
    assert surface["source_rows"] == committed_before
    assert surface["indexed_rows"] == committed_before
    assert surface["ready"] is True
    assert payload["message_indexable_count"] == payload["message_indexed_count"] == committed_before
    assert payload["coverage_pct"] == 100.0
    assert payload["messages_ready"] is True


@pytest.mark.parametrize(
    "diagnostic", ["cannot read '/opt/private space/例.json'", r"cannot read 'C:\Users\private space\例.json'"]
)
def test_returned_fts_failure_stays_unavailable_and_private_on_minimal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: str
) -> None:
    from unittest.mock import patch

    from polylogue.daemon import fts_status, status_snapshot

    index = tmp_path / "index.db"
    index.touch()

    def fail(*_args: object, **_kwargs: object) -> object:
        raise sqlite3.OperationalError(diagnostic)

    monkeypatch.setattr(fts_status, "open_readonly_connection", fail)
    # This test examines the returned failed-acquisition payload, after the
    # real collector has finished; deadline behavior has its own contract tests.
    import threading

    from polylogue.operations import status_protocol

    registry = fts_status._fts_readiness_registry(index)
    target_spec = registry.specs[0]
    completed = threading.Event()
    run_collector = status_protocol._run_collector

    def observe_completion(spec: object, attempt: object) -> None:
        run_collector(spec, attempt)  # type: ignore[arg-type]
        if spec is target_spec:
            completed.set()

    monkeypatch.setattr(status_protocol, "_run_collector", observe_completion)
    registry.request_refresh("fts_readiness")
    completed.wait()
    direct = fts_status.fts_readiness_info(index)
    with patch.object(status_snapshot, "resolve_active_index_path", lambda *_a, **_k: index):
        published = status_snapshot._minimal_status_payload()["fts_readiness"]
    assert isinstance(published, dict)
    for payload in (direct, published):
        assert payload["inspection_state"] == "unavailable", payload
        assert payload["messages_ready"] is False
        assert payload["coverage_pct"] is None
        error = str(payload["unavailable_reason"])
        assert "[redacted]" in error
        assert all(fragment not in error for fragment in ("/opt", "C:", "Users", "private space", "例.json"))
