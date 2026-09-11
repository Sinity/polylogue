"""Pinned direct-status producer conformance."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.operations.daemon_status import produce_direct_status
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.sqlite.archive_tiers.ops_write import record_schema_drift_sample
from tests.infra.archive_templates import bootstrap_archive_root


def test_direct_status_uses_the_pinned_archive_and_retains_legacy_sections(tmp_path: Path) -> None:
    """Mutation: reopen a tier or omit a legacy status block and this fails."""

    bootstrap_archive_root(tmp_path)
    with open_operation_read(tmp_path) as pinned:
        payload = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000)

    assert payload["archive_stats"]["total_sessions"] == 0
    assert payload["raw_materialization_readiness"]["available"] is True
    assert payload["raw_replay_backlog"]["available"] is True
    assert payload["archive_readiness"]["checked"] is False
    assert payload["sinex_publication"]["state"] == "not_observed"
    assert payload["assertion_candidate_queue"]["state"] == "not_observed"
    assert {"archive_tiers", "convergence", "schema_drift", "raw_frontier_integrity"} <= set(payload)


def test_direct_status_keeps_source_ops_and_embeddings_on_the_pinned_snapshot(tmp_path: Path) -> None:
    """Mutation: reopen any status tier after pinning and these durable facts change."""

    bootstrap_archive_root(tmp_path)
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
        payload = produce_direct_status(
            archive=pinned.archive,
            now_ms=1_700_000_000_002,
            include_archive_readiness=True,
        )

    assert payload["raw_parse_failures"] == 1
    assert payload["raw_failure_lifecycle_state"] == "blocked"
    assert payload["embedding_status"]["embedded_sessions"] == 1
    assert payload["schema_drift"]["origins"][0]["total"] == 1
    readiness = payload["raw_materialization_readiness"]
    assert readiness["raw_authority_parser_census"]["available"] is True
    assert readiness["raw_authority_frontier"] is None


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
        payload = produce_direct_status(archive=pinned.archive, now_ms=1_700_000_000_000, config=config)

    embeddings = payload["embedding_status"]
    assert embeddings["has_voyage_api_key"] is True
    assert embeddings["configured_model"] == "voyage-3-lite"
    assert embeddings["configured_dimension"] == 512
    assert embeddings["config_enabled"] is None
    assert embeddings["daemon_stage_enabled"] is None
    assert embeddings["monthly_cost_cap_usd"] is None
