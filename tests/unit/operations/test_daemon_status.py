"""Pinned direct-status producer conformance."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict, cast

import pytest

from polylogue.operations.daemon_status import _sqlite_maintenance, produce_direct_status
from polylogue.operations.operation_context import open_operation_read, prepare_operation_journals
from polylogue.storage.sqlite.archive_tiers.ops_write import record_schema_drift_sample
from tests.infra.archive_templates import bootstrap_archive_root


class _ArchiveStats(TypedDict):
    total_sessions: int
    total_messages: int


class _AvailableStatus(TypedDict):
    available: bool


class _StateStatus(TypedDict):
    state: str


class _ArchiveReadiness(TypedDict):
    checked: bool


class _TierStatus(TypedDict):
    table_counts: dict[str, int]


class _RawMaterializationStatus(TypedDict):
    available: bool
    raw_authority_parser_census: _AvailableStatus
    raw_authority_blocker_count: int


class _SchemaOriginStatus(TypedDict):
    total: int


class _SchemaDriftStatus(TypedDict):
    origins: list[_SchemaOriginStatus]


class _EmbeddingStatus(TypedDict):
    embedded_sessions: int
    embedded_messages: int
    has_voyage_api_key: bool | None
    configured_model: str | None
    configured_dimension: int | None
    config_enabled: bool | None
    daemon_stage_enabled: bool | None
    monthly_cost_cap_usd: float | None


class _StatusPayload(TypedDict):
    ok: bool
    archive_stats: _ArchiveStats
    raw_materialization_readiness: _RawMaterializationStatus
    raw_replay_backlog: _AvailableStatus
    archive_readiness: _ArchiveReadiness
    sinex_publication: _StateStatus
    assertion_candidate_queue: _StateStatus
    archive_tiers: dict[str, _TierStatus]
    raw_parse_failures: int
    raw_failure_lifecycle_state: str
    embedding_status: _EmbeddingStatus
    schema_drift: _SchemaDriftStatus


def test_direct_status_uses_the_pinned_archive_and_retains_legacy_sections(tmp_path: Path) -> None:
    """Mutation: reopen a tier or omit a legacy status block and this fails."""

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        payload = cast(_StatusPayload, produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000))

    assert payload["archive_stats"]["total_sessions"] == 0
    assert payload["raw_materialization_readiness"]["available"] is True
    assert payload["raw_replay_backlog"]["available"] is True
    assert payload["archive_readiness"]["checked"] is False
    assert payload["sinex_publication"]["state"] == "not_observed"
    assert payload["assertion_candidate_queue"]["state"] == "not_observed"
    assert payload["archive_tiers"]["audit"]["table_counts"] == {
        "operation_previews": 0,
        "operation_authorizations": 0,
        "operation_attempts": 0,
    }
    assert {"archive_tiers", "convergence", "schema_drift", "raw_frontier_integrity"} <= set(payload)


def test_direct_status_marks_missing_declared_relation_unavailable(tmp_path: Path) -> None:
    """A missing source table is not a measured empty count."""

    bootstrap_archive_root(tmp_path)
    with sqlite3.connect(tmp_path / "audit.db") as audit_conn:
        audit_conn.execute("DROP TABLE operation_attempts")
        audit_conn.commit()

    with open_operation_read(tmp_path) as pinned:
        payload = cast(_StatusPayload, produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000))

    audit_tier = payload["archive_tiers"]["audit"]
    assert payload["ok"] is False
    assert audit_tier["table_counts"]["operation_attempts"] is None
    precision = cast(dict[str, str], audit_tier["table_count_precision"])
    assert precision["operation_attempts"] == "missing"


def test_sqlite_maintenance_does_not_turn_absent_tier_into_zero(tmp_path: Path) -> None:
    """Maintenance metadata is explicitly unavailable when a tier is absent."""

    db = tmp_path / "index.db"
    with sqlite3.connect(db) as conn:
        maintenance = _sqlite_maintenance(conn)

    source = cast(dict[str, object], maintenance["tiers"])["source"]
    assert isinstance(source, dict)
    assert source["sqlite_stat1_rows"] is None
    assert source["planner_stats_present"] is None
    assert source["state"] == "unavailable"


def test_direct_status_keeps_source_ops_and_embeddings_on_the_pinned_snapshot(tmp_path: Path) -> None:
    """Mutation: reopen any status tier after pinning and these durable facts change."""

    bootstrap_archive_root(tmp_path)
    prepare_operation_journals(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, parse_error
            ) VALUES ('raw-first', 'codex-session', 'first', '/fixture/first.jsonl', ?, 1, 1, 'decode error')
            """,
            (b"x" * 32,),
        )
    with sqlite3.connect(tmp_path / "index.db") as index:
        index.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, content_hash, message_count)
            VALUES ('first', 'codex-session', 'raw-first', ?, 1)
            """,
            (b"y" * 32,),
        )
        session_id = str(index.execute("SELECT session_id FROM sessions WHERE raw_id = 'raw-first'").fetchone()[0])
    with sqlite3.connect(tmp_path / "embeddings.db") as embeddings:
        embeddings.execute(
            """
            INSERT INTO embedding_status (session_id, origin, message_count_embedded, needs_reindex)
            VALUES (?, 'codex-session', 1, 0)
            """,
            (session_id,),
        )
    with sqlite3.connect(tmp_path / "ops.db") as ops:
        record_schema_drift_sample(
            ops,
            origin="codex-session",
            element_kind="session_record",
            classification="field_changed",
            unseen_key_signature="fixture",
            native_id_example="first",
            raw_id="raw-first",
            observed_at_ms=1_700_000_000_000,
        )

    with open_operation_read(tmp_path) as pinned:
        # These writes happen after all attached operation snapshots have been
        # forced.  A status producer that opens ordinary root paths sees two.
        with sqlite3.connect(tmp_path / "source.db") as source:
            source.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms, parse_error
                ) VALUES ('raw-late', 'codex-session', 'late', '/fixture/late.jsonl', ?, 1, 2, 'decode error')
                """,
                (b"z" * 32,),
            )
        with sqlite3.connect(tmp_path / "ops.db") as ops:
            record_schema_drift_sample(
                ops,
                origin="codex-session",
                element_kind="session_record",
                classification="field_changed",
                unseen_key_signature="late",
                native_id_example="late",
                raw_id="raw-late",
                observed_at_ms=1_700_000_000_001,
            )
        with sqlite3.connect(tmp_path / "embeddings.db") as embeddings:
            embeddings.execute(
                """
                INSERT INTO embedding_status (session_id, origin, message_count_embedded, needs_reindex)
                VALUES ('late-session', 'codex-session', 9, 0)
                """
            )
        payload = cast(
            _StatusPayload,
            produce_direct_status(
                archive=pinned.archive,
                now_ms=1_700_000_000_002,
                include_archive_readiness=True,
            ),
        )

    assert payload["raw_parse_failures"] == 1
    assert payload["raw_failure_lifecycle_state"] == "blocked"
    # `late-session` has no `index.sessions` row, so the authoritative rollup
    # never sees it: since #5027 readiness is classified over desired message
    # membership in the pinned index tier, not over `embedding_status`
    # telemetry. Both numbers below now come from that one CTE, so the old
    # (0 sessions, 1 message) pair is unreachable by construction -- it was a
    # pre-#5027 shape where the two came from different sources.
    #
    # The seeded session has no `messages` rows at all, so it is a valid-empty
    # partition (required == valid == 0) and counts as embedded, while
    # `embedded_messages` is 0 because no vector, ref, or meta row exists
    # anywhere in the fixture. A bare `embedding_status` row still certifies
    # nothing -- the invariant the old comment wanted is enforced more
    # strictly now, not less.
    assert payload["embedding_status"]["embedded_sessions"] == 1
    assert payload["embedding_status"]["embedded_messages"] == 0
    assert payload["schema_drift"]["origins"][0]["total"] == 1
    readiness = payload["raw_materialization_readiness"]
    assert readiness["raw_authority_parser_census"]["available"] is True
    assert readiness["raw_authority_blocker_count"] == 0


def test_embedding_status_preserves_supplied_key_without_guessing_missing_settings(tmp_path: Path) -> None:
    """Mutation: read PolylogueConfig-only fields from Config and the key is lost."""
    from polylogue.config import Config, IndexConfig

    bootstrap_archive_root(tmp_path)
    config = Config(
        archive_root=tmp_path,
        render_root=tmp_path / "render",
        sources=[],
        index_config=IndexConfig(voyage_api_key="synthetic-voyage-key"),
        embedding_model="voyage-3-lite",
        embedding_dimension=512,
    )
    with open_operation_read(tmp_path) as pinned:
        payload = cast(
            _StatusPayload,
            produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000, config=config),
        )

    embeddings = payload["embedding_status"]
    assert embeddings["has_voyage_api_key"] is True
    assert embeddings["configured_model"] == "voyage-3-lite"
    assert embeddings["configured_dimension"] == 512
    assert embeddings["config_enabled"] is None
    assert embeddings["daemon_stage_enabled"] is None
    assert embeddings["monthly_cost_cap_usd"] is None


def test_active_archive_root_match_is_compared_not_asserted(tmp_path: Path) -> None:
    """polylogue-bu47u: the direct payload hardcoded ``True`` for this comparison.

    The honest comparison already existed in ``cli/commands/paths.py``; the
    status producer published a literal instead, so an active index served from
    somewhere other than the configured root still reported as matching.

    Anti-vacuity: restore ``"active_archive_root_matches_configured": True``
    and the redirected case below stays True, so this fails. The first
    assertion fails if the comparison is inverted or always False.
    """

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        matching = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000)
        assert matching["active_archive_root_matches_configured"] is True
        assert matching["active_archive_root"] == str(tmp_path)

        # The store's own declared active index moves to a promoted generation
        # directory inside the same configured root.
        pinned.archive.index_db_path = tmp_path / "generations" / "g2" / "index.db"
        redirected = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000)

    assert redirected["active_archive_root_matches_configured"] is False
    assert redirected["active_archive_root"] == str(tmp_path / "generations" / "g2")
    assert redirected["archive_root"] == str(tmp_path)


def test_ops_workload_throughput_rate_uses_the_advertised_wall_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """polylogue-i6p4x: concurrent attempts must not deflate files_per_second.

    Four attempts each run the full 60s wall window and parse 60 files apiece,
    so the archive really ingested 240 files in 60 seconds: 4.0 files/s. The
    old denominator summed per-attempt busy time (240s) and reported 1.0 --
    understated by exactly the concurrency factor, while the sibling
    ``window_minutes`` key advertised a wall window.

    Anti-vacuity: restore ``SUM(finished_at_ms - started_at_ms)`` as the
    denominator and this reports 1.0 instead of 4.0.
    """

    from polylogue.operations import status_workload
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_ingest_attempt
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    monkeypatch.setattr(status_workload, "_WORKLOAD_THROUGHPUT_WINDOW_MS", 60_000)

    ops_db = tmp_path / "ops.db"
    initialize_archive_database(ops_db, ArchiveTier.OPS)
    now_ms = 1_800_000_000_000
    conn = sqlite3.connect(ops_db)
    try:
        for index in range(4):
            record_ingest_attempt(
                conn,
                status="completed",
                source_path=f"/synthetic/source-{index}.jsonl",
                started_at_ms=now_ms - 60_000,
                finished_at_ms=now_ms,
                parsed_raw_count=60,
            )
        conn.commit()

        payload = status_workload.ops_workload_status_from_connection(conn, now_ms=now_ms, schema="main")
    finally:
        conn.close()

    throughput = cast(dict[str, object], payload["throughput"])
    assert throughput["window_minutes"] == 1
    assert throughput["files"] == 240
    assert throughput["files_per_second"] == 4.0
