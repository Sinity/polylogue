"""Fast-path coverage for ``polylogue ops embed status``."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.embed import embed_command
from polylogue.config import PolylogueConfig
from polylogue.storage.embeddings import status_payload as status_payload_mod
from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingSourceDigest, vector_derivation_hash
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    ArchiveEmbeddingWrite,
    begin_embedding_attempt,
    complete_embedding_attempt_success,
    record_embedding_failure,
    resolve_embedding_failure,
)
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from tests.infra.embedding_config import embedding_config


def _cfg(*, embedding_enabled: bool, voyage_api_key: str | None) -> PolylogueConfig:
    """A real ``PolylogueConfig``, not a duck-typed stand-in.

    ``embedding_status_settings_from_config`` dispatches nominally over
    ``PolylogueConfig``/``Config`` and refuses anything else; see
    ``tests/infra/embedding_config.py`` for why structural matching cannot
    stand in for it.
    """

    return embedding_config(
        embedding_enabled=embedding_enabled,
        voyage_api_key=voyage_api_key,
        embedding_model="voyage-4",
        embedding_dimension=1024,
        embedding_max_cost_usd=5.0,
    )


def _env(db_path: Path) -> Any:
    env = MagicMock()
    env.config.db_path = db_path
    env.config.archive_root = db_path.parent
    return env


def _stamp_tier(path: Path, tier_name: str) -> None:
    """Stamp a seeded tier file with the runtime's expected ``user_version``.

    A derived tier whose version is 0 is schema skew, not an empty archive:
    the reader refuses with ``SchemaSkewError`` before looking at any row.
    ``_run_status`` already does this for the index; a seeded ops or
    embeddings tier needs the same or it is refused rather than read.
    """
    from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    version = ARCHIVE_VERSION_BY_TIER[ArchiveTier(tier_name)]
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {version}")


def _index_path(db_path: Path) -> Path:
    """Resolve the tier the production fast path reads, as ``_run_status`` does."""
    return db_path if db_path.name == "index.db" else db_path.with_name("index.db")


def _seed_archive_without_embedding_ledgers(
    db_path: Path, *, vec_table: bool = False, at_index_tier: bool = True
) -> None:
    # Seed the INDEX tier, which is where the production fast path looks:
    # ``_locate_archive`` resolves ``ArchiveLocation.resolve(parent)`` and
    # requires a ``sessions`` table in ``active_index_path``. These helpers
    # predate the split-tier layout and wrote to ``archive.db``, so every
    # assertion here read an archive production considered empty -- the status
    # came back ``empty``/0 rather than the seeded 2 sessions. Resolve the same
    # way ``_run_status`` does so the fixture and the route agree on one file.
    # ``at_index_tier=False`` seeds the given file verbatim, for the one test
    # that needs a non-index anchor file to exist alongside a real index.db.
    db_path = _index_path(db_path) if at_index_tier else db_path
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.executemany(
            "INSERT INTO messages (message_id, session_id, content_hash) VALUES (?, ?, ?)",
            [("msg-1", "conv-1", "h1"), ("msg-2", "conv-2", "h2")],
        )
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        if vec_table:
            conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.commit()


# v4 (polylogue-q88p): distinct, >=20-char prose per message so each message's
# vector_derivation_hash -- computed by archive_embeddable_messages_relation via
# the registered SQL function from exactly this text -- is real and unique,
# matching what production actually sends to the embedder.
_COMPLETE_TEXT = "authored prose for the complete session message, long enough to embed"
_PENDING_TEXT_1 = "authored prose for the first pending session message, long enough"
_PENDING_TEXT_2 = "authored prose for the second pending session message, long enough"


def _pending_session_source_hash() -> bytes:
    """The exact session-level source_hash `codex-session:pending`'s 2 real
    messages produce -- hash VALUES only, sorted, matching the production
    aggregate (`_archive_embedding_source_hash_from_pairs`)."""
    hashes = sorted(
        vector_derivation_hash(model="voyage-4", input_text=text) for text in (_PENDING_TEXT_1, _PENDING_TEXT_2)
    )
    digest = EmbeddingSourceDigest()
    for value in hashes:
        digest.update(value)
    return digest.digest()


def _message_content_hash_stub(message_id: str) -> bytes:
    """A deterministic 32-byte stand-in for ``messages.content_hash``.

    This fixture builds index.db by hand (the status fast path reads only
    sessions/messages), so nothing here computes the canonical message
    semantic hash. What matters for v6 is that the value is a real 32-byte
    hash and that the *same* value reaches ``message_embedding_refs.
    message_content_hash`` through the production embedding write, because
    inspection compares those two columns for equality.
    """
    return hashlib.sha256(message_id.encode("utf-8")).digest()


def _deterministic_vector(seed: str) -> list[float]:
    """A real 1024-float vector, not a placeholder the write path would reject."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return [((digest[i % len(digest)] + i) % 256) / 256.0 for i in range(EMBEDDING_DIMENSION)]


def _seed_archive_file_set_from_archive_tiers(index_db: Path) -> None:
    """Build a minimal but real v6-shaped index.db + embeddings.db pair.

    embeddings.db is created from the production DDL
    (``initialize_archive_database(..., ArchiveTier.EMBEDDINGS)``) and its one
    embedded session is published through the production write route --
    ``begin_embedding_attempt`` then ``complete_embedding_attempt_success`` --
    so ``message_embeddings`` is the real ``vec0`` virtual table holding a real
    1024-dimension vector, ``message_embeddings_meta`` carries the complete
    recipe/output-contract identity, and ``message_embedding_refs`` carries the
    v6 ``message_content_hash`` column. Nothing here hand-inserts an embedding
    row; a fabricated row set could drift from what production writes.

    Anti-vacuity: the v6 inspection predicate
    (``_authoritative_archive_embedding_state`` in status_payload.py) requires, per
    required message, the current ref *and* the current message semantic hash
    *and* the complete recipe identity *and*
    ``EXISTS (SELECT 1 FROM message_embeddings ...)``. Drop any one of those
    conjuncts -- in particular the physical-vector EXISTS -- and
    ``test_status_json_reports_metadata_only_row_as_not_ready`` below goes red,
    because that test deletes only the vector and keeps every metadata row.
    """
    embeddings_db = index_db.with_name("embeddings.db")
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 1);
            INSERT INTO sessions VALUES ('codex-session:pending', 2);
            """
        )
        for message_id, session_id, text in (
            ("codex-session:complete:m1", "codex-session:complete", _COMPLETE_TEXT),
            ("codex-session:pending:m1", "codex-session:pending", _PENDING_TEXT_1),
            ("codex-session:pending:m2", "codex-session:pending", _PENDING_TEXT_2),
        ):
            conn.execute(
                "INSERT INTO messages (message_id, session_id, text, content_hash) VALUES (?, ?, ?, ?)",
                (message_id, session_id, text, _message_content_hash_stub(message_id)),
            )
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        conn.commit()

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)

    complete_message_id = "codex-session:complete:m1"
    recipe = EmbeddingRecipe.current(model="voyage-4", dimensions=EMBEDDING_DIMENSION)
    complete_hash = vector_derivation_hash(model="voyage-4", input_text=_COMPLETE_TEXT)
    source_digest = EmbeddingSourceDigest()
    source_digest.update(complete_hash)

    conn = sqlite3.connect(embeddings_db)
    try:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            # A stubbed vector would defeat the point of this fixture: the
            # readiness predicate it exercises is precisely "a physical vector
            # exists at this address".
            pytest.fail(f"sqlite-vec is required to build a v6 embedding fixture: {error}")
        attempt = begin_embedding_attempt(
            conn,
            session_id="codex-session:complete",
            origin="codex-session",
            source_hash=source_digest.digest(),
            recipe=recipe,
            started_at_ms=1767225700000,
        )
        published = complete_embedding_attempt_success(
            conn,
            attempt=attempt,
            writes=[
                ArchiveEmbeddingWrite(
                    message_id=complete_message_id,
                    session_id="codex-session:complete",
                    origin="codex-session",
                    embedding=_deterministic_vector(complete_message_id),
                    model="voyage-4",
                    embedded_at_ms=1767225700000,
                    vector_derivation_hash=complete_hash,
                    message_content_hash=_message_content_hash_stub(complete_message_id),
                    recipe_hash=recipe.recipe_hash,
                    output_contract_hash=recipe.output_contract_hash,
                    generation=attempt.generation,
                )
            ],
            completed_at_ms=1767225700000,
        )
        assert published
    finally:
        conn.close()


def _open_embeddings(path: Path) -> sqlite3.Connection:
    """Open embeddings.db with the vector extension loaded.

    ``message_embeddings`` is a real ``vec0`` virtual table in v6, so a plain
    connection fails with "no such module: vec0" on any statement that names
    it -- including DDL-free DELETEs.
    """
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.fail(f"sqlite-vec is required to read a v6 embedding fixture: {error}")
    return conn


def _payload(result_output: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(result_output))


def _run_status(db_path: Path, *args: str, cfg: PolylogueConfig | None = None) -> dict[str, Any]:
    index_db = db_path if db_path.name == "index.db" else db_path.with_name("index.db")
    if index_db.exists():
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        with sqlite3.connect(index_db) as conn:
            if int(conn.execute("PRAGMA user_version").fetchone()[0]) == 0:
                conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=cfg or _cfg(embedding_enabled=False, voyage_api_key=None),
    ):
        result = runner.invoke(
            embed_command,
            ["status", "--format", "json", *args],
            obj=_env(db_path),
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    return _payload(result.output)


def _run_status_text(db_path: Path, *, detail: bool = False, cfg: PolylogueConfig | None = None) -> str:
    index_db = db_path if db_path.name == "index.db" else db_path.with_name("index.db")
    if index_db.exists():
        from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        with sqlite3.connect(index_db) as conn:
            if int(conn.execute("PRAGMA user_version").fetchone()[0]) == 0:
                conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=cfg or _cfg(embedding_enabled=False, voyage_api_key=None),
    ):
        result = runner.invoke(
            embed_command,
            ["status", *(["--detail"] if detail else [])],
            obj=_env(db_path),
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    return str(result.output)


def test_status_json_fast_path_handles_absent_embedding_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path)

    assert payload["status"] == "none"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 0
    assert payload["pending_sessions"] == 2
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["retrieval_bands"] == {}


def test_status_json_reads_archive_file_set_from_archive_index(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["embedded_messages"] == 1
    assert payload["pending_messages"] == 2
    assert payload["pending_messages_exact"] is True
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["stale_messages"] == 0
    assert payload["retrieval_ready"] is True
    assert payload["freshness_status"] == "partial"
    assert payload["embedding_coverage_percent"] == 50.0
    assert payload["embedding_coverage_basis"] == "sessions"
    assert payload["message_coverage_percent"] == 33.3
    assert payload["embedding_models"] == {"voyage-4": 1}
    assert payload["embedding_dimensions"] == {"1024": 1} or payload["embedding_dimensions"] == {1024: 1}


def test_status_json_reports_archive_embedding_metadata_without_detail(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["embedding_models"] == {}
    assert payload["embedding_dimensions"] == {}
    assert payload["oldest_embedded_at"] is None
    assert payload["newest_embedded_at"] is None


def test_status_detail_exposes_bounded_terminal_failure_resolution(tmp_path: Path) -> None:
    """Status must name an actionable terminal row rather than only its aggregate.

    Anti-vacuity: omitting the text renderer leaves the exact failure id and
    resolution command unreachable to the operator despite JSON detail.
    """
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    with sqlite3.connect(index_db.with_name("embeddings.db")) as conn:
        conn.execute(
            """
            INSERT INTO embedding_failures (
                failure_id, session_id, origin, message_refs_json, provider, model,
                error_class, error_message, retryable, lifecycle_state,
                created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
                resolution_note, superseded_by
            ) VALUES (
                'embedding-failure:terminal', 'codex-session:pending', 'codex-session',
                '[\"codex-session:pending:m1\"]', 'voyage', 'voyage-4', 'provider_http_400',
                'Embedding generation failed: HTTP 400', 0, 'terminal', 1800000000000, 1800000000000,
                NULL, NULL, NULL, NULL
            )
            """
        )

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["failure_count"] == 1
    assert payload["terminal_failure_count"] == 1
    assert payload["retryable_failure_count"] == 0
    assert payload["failure_details"] == [
        {
            "failure_id": "embedding-failure:terminal",
            "session_id": "codex-session:pending",
            "origin": "codex-session",
            "message_refs": ["codex-session:pending:m1"],
            "provider": "voyage",
            "model": "voyage-4",
            "error_class": "provider_http_400",
            "error_message": "Embedding generation failed: HTTP 400",
            "retryable": False,
            "lifecycle_state": "terminal",
            "created_at": "2027-01-15T08:00:00+00:00",
            "updated_at": "2027-01-15T08:00:00+00:00",
            "resolution_action": None,
            "supported_actions": ["acknowledge", "requeue", "supersede"],
            "resolution_command": (
                "polylogue ops embed resolve-failure embedding-failure:terminal"
                " --action <acknowledge|requeue|supersede> --yes"
            ),
        }
    ]
    text = _run_status_text(db_anchor, detail=True, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))
    assert "embedding-failure:terminal: terminal" in text
    assert "refs: codex-session:pending:m1" in text
    assert "resolve: polylogue ops embed resolve-failure embedding-failure:terminal" in text


def test_resolve_failure_cli_requeues_terminal_failure(tmp_path: Path) -> None:
    """The operator command must mutate the failure ledger and retry status.

    Anti-vacuity: replacing the Click command with output-only formatting, or
    removing its call to ``resolve_embedding_failure``, leaves the terminal row
    active and the session excluded from a future embedding pass.
    """
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    embeddings_db = index_db.with_name("embeddings.db")
    with sqlite3.connect(embeddings_db) as conn:
        # embedding_failures now comes from the production EMBEDDINGS_DDL; only
        # the rows this test needs are seeded here.
        conn.executescript(
            """
            INSERT INTO embedding_status (session_id, origin, message_count_embedded, needs_reindex, error_message)
            VALUES ('codex-session:pending', 'codex-session', 0, 0, 'Embedding generation failed: HTTP 400');
            INSERT INTO embedding_failures (
                failure_id, session_id, origin, message_refs_json, provider, model,
                error_class, error_message, retryable, lifecycle_state,
                created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
                resolution_note, superseded_by
            ) VALUES (
                'embedding-failure:terminal', 'codex-session:pending', 'codex-session',
                '["codex-session:pending:m1"]', 'voyage', 'voyage-4', 'provider_http_400',
                'Embedding generation failed: HTTP 400', 0, 'terminal', 1800000000000, 1800000000000,
                NULL, NULL, NULL, NULL
            );
            """
        )

    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    result = runner.invoke(
        embed_command,
        [
            "resolve-failure",
            "embedding-failure:terminal",
            "--action",
            "requeue",
            "--yes",
            "--format",
            "json",
        ],
        obj=_env(db_anchor),
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert _payload(result.output) == {
        "failure_id": "embedding-failure:terminal",
        "lifecycle_state": "resolved",
        "resolution_action": "requeue",
        "resolution_note": None,
        "session_id": "codex-session:pending",
        "superseded_by": None,
    }
    with sqlite3.connect(embeddings_db) as conn:
        assert conn.execute(
            "SELECT lifecycle_state FROM embedding_failures WHERE failure_id = 'embedding-failure:terminal'"
        ).fetchone() == ("resolved",)
        assert conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = 'codex-session:pending'"
        ).fetchone() == (1, None)


def test_status_excludes_acknowledged_terminal_failure_from_retry_backlog(tmp_path: Path) -> None:
    """An acknowledgement clears critical debt without falsifying coverage.

    Anti-vacuity: removing the blocked-session lifecycle query makes detail
    status report this unembedded session as complete, while summary status
    presents it as a retryable backlog even though the writer excludes it.
    """
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    embeddings_db = index_db.with_name("embeddings.db")
    with sqlite3.connect(index_db) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = 'codex-session:complete'")
        conn.execute("DELETE FROM sessions WHERE session_id = 'codex-session:complete'")
    with _open_embeddings(embeddings_db) as conn:
        conn.execute(
            """
            DELETE FROM message_embeddings
            WHERE vector_derivation_hash = (
                SELECT lower(hex(vector_derivation_hash)) FROM message_embedding_refs
                WHERE message_id = 'codex-session:complete:m1'
            )
            """
        )
        # message_embeddings_meta is content-addressed (vector_derivation_hash-
        # keyed, v4) -- resolve via the ref, then delete both.
        conn.execute(
            """
            DELETE FROM message_embeddings_meta
            WHERE vector_derivation_hash = (
                SELECT vector_derivation_hash FROM message_embedding_refs WHERE message_id = 'codex-session:complete:m1'
            )
            """
        )
        conn.execute("DELETE FROM message_embedding_refs WHERE message_id = 'codex-session:complete:m1'")
        conn.execute("DELETE FROM embedding_status WHERE session_id = 'codex-session:complete'")
    with sqlite3.connect(embeddings_db) as conn:
        # "blocked" is read off embedding_derivation_state.attempt_state =
        # 'failed_terminal' (the unified freshness key), not merely an
        # embedding_status row with an error_message -- go through the real
        # begin_embedding_attempt/record_embedding_failure write path so this
        # fixture's terminal failure is one the modern predicate actually
        # recognizes, exactly like a real provider-rejected session would be.
        attempt = begin_embedding_attempt(
            conn,
            session_id="codex-session:pending",
            origin="codex-session",
            source_hash=_pending_session_source_hash(),
            recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
            started_at_ms=1_800_000_000_000,
        )
        failure = record_embedding_failure(
            conn,
            session_id="codex-session:pending",
            origin="codex-session",
            message_refs=("codex-session:pending:m1",),
            provider="voyage",
            model="voyage-4",
            error_class="provider_http_400",
            error_message="Embedding generation failed: HTTP 400",
            retryable=False,
            occurred_at_ms=1_800_000_000_000,
            attempt=attempt,
        )
        resolve_embedding_failure(
            conn,
            failure_id=failure.failure_id,
            action="acknowledge",
            resolved_at_ms=1_800_000_001_000,
        )

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["failure_count"] == 0
    assert payload["terminal_failure_count"] == 0
    assert payload["blocked_sessions"] == 1
    assert payload["pending_sessions"] == 0
    assert payload["pending_messages"] == 0
    assert payload["embedding_coverage_percent"] == 0.0
    assert payload["status"] == "partial"
    assert payload["next_action"] == {
        "code": "acknowledged_terminal_exclusions",
        "command": None,
        "reason": (
            "Some sessions have acknowledged or superseded terminal embedding failures; "
            "they are retained as audit evidence but excluded from automatic retry."
        ),
    }


def test_status_json_detail_does_not_derive_coverage_from_analyzed_prose_estimate(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE INDEX idx_messages_embedding_prose
            ON messages(session_id, message_id)
            WHERE message_type = 'message'
              AND role IN ('user', 'assistant')
              AND material_origin IN ('human_authored', 'assistant_authored')
              AND word_count > 0;
            ANALYZE idx_messages_embedding_prose;
            """
        )

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is False
    assert payload["message_coverage_percent"] is None


def test_status_json_does_not_report_over_100_percent_from_retained_embedding_rows(tmp_path: Path) -> None:
    """An inflated ``message_count_embedded`` ledger entry must never push
    coverage past 100%.

    v6: readiness is inspected directly from desired message membership,
    current refs, complete recipe metadata and physical vectors.
    ``embedding_status.message_count_embedded`` is attempt telemetry and is not
    an input at all, so an inflated counter -- or an extra ledger row for a
    session that does not exist -- cannot move coverage. That is what removes
    the >100% failure mode: there is no path where a ledger number can present
    a session as embedded-and-therefore-covered when no vector backs it.
    """
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE INDEX idx_messages_embedding_prose
            ON messages(session_id, message_id)
            WHERE message_type = 'message'
              AND role IN ('user', 'assistant')
              AND material_origin IN ('human_authored', 'assistant_authored')
              AND word_count > 0;
            ANALYZE idx_messages_embedding_prose;
            """
        )
        with sqlite3.connect(tmp_path / "embeddings.db") as conn:
            # A content-addressed vector/meta row with no surviving ref -- e.g. the
            # message that referenced it was removed by an index rebuild. v4 never
            # deletes these (reconcile.py module docstring); status must not let a
            # retained, refless row inflate coverage past 100%.
            retained_recipe = EmbeddingRecipe.current(model="voyage-4", dimensions=1024)
            conn.execute(
                "INSERT INTO message_embeddings_meta VALUES (?, 'voyage-4', 1024, 1767225700000, ?, ?)",
                (b"\xff" * 32, retained_recipe.recipe_hash, retained_recipe.output_contract_hash),
            )
        conn.execute(
            "UPDATE embedding_status SET message_count_embedded = 4 WHERE session_id = 'codex-session:complete'"
        )
        conn.executemany(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, needs_reindex, error_message
            ) VALUES (?, 'codex-session', ?, 0, ?)
            """,
            [
                ("orphaned-clean-status", 7, None),
                ("orphaned-error-status", 0, "Embedding generation failed: HTTP 400"),
            ],
        )

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    # codex-session:complete's inflated message_count_embedded (4) no longer
    # matches its real eligible count (1) or its embedding_derivation_state
    # row's message_count (1), so the exact predicate demotes it to pending
    # instead of trusting the stale number -- coverage lands at 0%, never over
    # 100%.
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["embedding_coverage_percent"] == 50.0
    assert payload["embedded_messages"] == 1
    assert payload["failure_count"] == 0
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is False
    assert payload["message_coverage_percent"] is None


def test_status_json_default_uses_bounded_exact_archive_session_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    observed: dict[str, object] = {}

    def fake_exact_session_state(*args: object, **kwargs: object) -> status_payload_mod.ArchiveEmbeddingStateProbe:
        observed.update(kwargs)
        return status_payload_mod.ArchiveEmbeddingStateProbe(counts=(1, 1, 1, 0))

    monkeypatch.setattr(
        status_payload_mod,
        "_authoritative_archive_embedding_state",
        fake_exact_session_state,
    )

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    # v6 inspects refs + recipe metadata + physical vectors together; the
    # session status ledger is no longer an input to readiness.
    assert observed["refs_table"] == "embeddings.message_embedding_refs"
    assert observed["meta_table"] == "embeddings.message_embeddings_meta"
    assert observed["vectors_table"] == "embeddings.message_embeddings"
    assert observed["timeout_ms"] == status_payload_mod.METADATA_SUMMARY_TIMEOUT_MS
    assert payload["status"] == "partial"
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False


def test_status_json_default_skips_embedding_metadata_summary_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    real_rows_with_timeout = status_payload_mod._rows_with_timeout

    def reject_metadata_summary_rows(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> list[sqlite3.Row | tuple[object, ...]] | None:
        # The metadata *summaries* are the per-model / per-dimension aggregates.
        # The v6 readiness inspection also joins message_embeddings_meta, but it
        # is a bounded per-session predicate, not a summary scan.
        if "GROUP BY model" in sql or "GROUP BY dimension" in sql:
            raise AssertionError("default embedding status must not scan metadata summaries")
        return real_rows_with_timeout(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_rows_with_timeout", reject_metadata_summary_rows)

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["embedding_models"] == {}
    assert payload["embedding_dimensions"] == {}


def test_status_json_detail_uses_uniform_metadata_probe_when_grouping_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    real_rows_with_timeout = status_payload_mod._rows_with_timeout

    def fake_rows_with_timeout(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> list[sqlite3.Row | tuple[object, ...]] | None:
        if "GROUP BY model" in sql or "GROUP BY dimension" in sql:
            return None
        return real_rows_with_timeout(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_rows_with_timeout", fake_rows_with_timeout)

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["embedding_models"] == {"voyage-4": 1}
    assert payload["embedding_dimensions"] == {"1024": 1} or payload["embedding_dimensions"] == {1024: 1}


def test_status_json_detail_keeps_the_metadata_summary_that_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One timed-out metadata lane must not discard the other lane's exact counts.

    The archive holds two models at one uniform dimension, so the uniform
    single-value probe legitimately proves nothing and returns empty. The
    per-dimension GROUP BY completes and is exact.

    Anti-vacuity: restore the unconditional
    ``model_counts, dimension_counts = _uniform_embedding_metadata_counts(...)``
    assignment and this goes red with ``embedding_dimensions == {}`` for an
    archive whose dimension histogram was measured exactly.
    """
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    with sqlite3.connect(tmp_path / "embeddings.db") as conn:
        conn.execute(
            """
            INSERT INTO message_embeddings_meta (
                vector_derivation_hash, model, dimension, embedded_at_ms, recipe_hash, output_contract_hash
            )
            SELECT randomblob(32), 'voyage-3', dimension, embedded_at_ms, recipe_hash, output_contract_hash
            FROM message_embeddings_meta
            LIMIT 1
            """
        )
        conn.commit()

    real_rows_with_timeout = status_payload_mod._rows_with_timeout

    def time_out_only_the_model_summary(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> list[sqlite3.Row | tuple[object, ...]] | None:
        if "GROUP BY model" in sql:
            return None
        return real_rows_with_timeout(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_rows_with_timeout", time_out_only_the_model_summary)

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["embedding_models"] == {}
    assert payload["embedding_dimensions"] in ({"1024": 2}, {1024: 2})


def test_status_json_refuses_to_certify_from_a_legacy_status_ledger(tmp_path: Path) -> None:
    """A clean session ledger over a pre-v4 metadata shape certifies nothing.

    This archive carries the old ``message_id``-keyed ``message_embeddings_meta``
    and a clean ``embedding_status`` row claiming one embedded message, but no
    ``message_embedding_refs`` with current message semantics and no physical
    vectors. Under v6 readiness is inspected -- current ref, current message
    content hash, complete recipe identity, and an existing vector -- so an
    uninspectable archive reports coverage as *unknown* rather than trusting
    the ledger. (This test previously asserted the ledger-trusting behavior,
    which v6 deliberately removed, and then a measured zero, which is the
    opposite lie.)

    Anti-vacuity: make the payload fall back to ``embedding_status.
    message_count_embedded`` when inspection is unavailable and this goes red
    with ``embedded_sessions == 1`` for an archive that holds no vector at all;
    collapse the unmeasurable state back to zero and it goes red on the
    ``is None`` assertion.
    """
    index_db = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                authored_user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 20, 20);
            INSERT INTO sessions VALUES ('codex-session:pending', 3, 1);
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:complete:m1', 'codex-session:complete', x'01');
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:pending:m1', 'codex-session:pending', x'02');
            """
        )
    with sqlite3.connect(embeddings_db) as conn:
        conn.executescript(
            """
            CREATE TABLE message_embeddings_meta (
                message_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedded_at_ms INTEGER NOT NULL,
                content_hash BLOB,
                needs_reindex INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO embedding_status VALUES ('codex-session:complete', 'codex-session', 1, 0, NULL);
            """
        )

    payload = _run_status(tmp_path / "custom.sqlite", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["coverage_measurable"] is False
    assert payload["status"] == "unknown"
    assert payload["embedded_sessions"] is None
    assert payload["pending_sessions"] is None
    assert payload["embedding_coverage_percent"] is None
    assert payload["retrieval_ready"] is False


def test_status_json_detail_falls_back_when_exact_pending_count_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    original_scalar = status_payload_mod._scalar_int_with_timeout

    def fake_scalar_int_with_timeout(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> int | None:
        # The exact pending-message count is the v6 per-message staleness
        # predicate: current ref, current message semantics, complete recipe,
        # and a physical vector.
        if "r.message_content_hash IS NOT m.content_hash" in sql:
            return None
        return original_scalar(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_scalar_int_with_timeout", fake_scalar_int_with_timeout)

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["pending_sessions"] == 1
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["message_coverage_percent"] == 33.3
    assert payload["total_estimated_cost_usd"] is None
    assert payload["retrieval_ready"] is True


def test_status_json_detail_falls_back_when_exact_session_state_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    def unavailable_session_state(*args: object, **kwargs: object) -> status_payload_mod.ArchiveEmbeddingStateProbe:
        # The v6 inspection cannot certify itself within its timeout; the
        # caller must report that it does not know, not guess.
        return status_payload_mod.ArchiveEmbeddingStateProbe(counts=None, reason="readiness_inspection_timeout")

    monkeypatch.setattr(
        status_payload_mod,
        "_authoritative_archive_embedding_state",
        unavailable_session_state,
    )

    payload = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    # Nothing is *known* when readiness cannot be inspected. The payload must
    # neither trust the session ledger's embedded count nor report a measured
    # zero -- reporting zero for an uninspectable archive prescribes a paid
    # regeneration of vectors that may be present and intact.
    #
    # Anti-vacuity: have the unmeasurable branch fall back to
    # ``embedding_status.message_count_embedded`` and this goes red with
    # ``embedded_sessions == 1``; collapse it back to a measured zero and it
    # goes red on the ``is None`` assertions below.
    assert payload["coverage_measurable"] is False
    assert payload["coverage_unmeasurable_reason"] == "readiness_inspection_timeout"
    assert payload["status"] == "unknown"
    assert payload["embedded_sessions"] is None
    assert payload["pending_sessions"] is None
    assert payload["next_action"]["code"] == "coverage_unmeasurable"
    assert payload["retrieval_ready"] is False
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True


def test_status_text_detail_does_not_claim_zero_cost_when_exact_pending_count_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    original_scalar = status_payload_mod._scalar_int_with_timeout

    def fake_scalar_int_with_timeout(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> int | None:
        # The exact pending-message count is the v6 per-message staleness
        # predicate: current ref, current message semantics, complete recipe,
        # and a physical vector.
        if "r.message_content_hash IS NOT m.content_hash" in sql:
            return None
        return original_scalar(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_scalar_int_with_timeout", fake_scalar_int_with_timeout)

    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=_cfg(embedding_enabled=True, voyage_api_key="vk-live"),
    ):
        result = runner.invoke(
            embed_command,
            ["status", "--detail"],
            obj=_env(db_anchor),
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Pending:              1 convs, msgs not calculated" in result.output
    assert "Session coverage:     50.0%" in result.output
    assert "Message coverage:     33.3% of 3 candidate prose msgs" in result.output
    assert "Estimated total cost: unknown" in result.output
    assert "use --detail" not in result.output


def test_status_json_reports_manual_backfill_when_config_disabled_but_partial(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["retrieval_ready"] is True
    assert payload["config_enabled"] is False
    assert payload["daemon_stage_enabled"] is False
    assert payload["next_action"] == {
        "code": "continue_backfill",
        "command": "polylogue ops embed backfill --yes --max-sessions 10",
        "reason": (
            "Manual embedding coverage exists, but daemon convergence is disabled; "
            "continue bounded backfill or enable daemon catch-up."
        ),
    }


def test_status_json_reads_latest_catchup_from_ops_db(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="v1-run",
            status="completed",
            started_at_ms=1_767_225_700_000,
            finished_at_ms=1_767_225_705_000,
            scanned_sessions=2,
            embedded_sessions=2,
            error_count=0,
            embedded_messages=4,
            estimated_cost_usd=0.001,
        )

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    assert latest["run_id"] == "v1-run"
    assert latest["status"] == "completed"
    assert latest["processed_sessions"] == 2
    assert latest["embedded_sessions"] == 2
    assert latest["error_count"] == 0
    assert latest["embedded_messages"] == 4
    assert latest["estimated_cost_usd"] == 0.001
    assert payload["latest_material_catchup_run"] == latest


def test_archive_backfill_stop_persists_canonical_interrupted_status(tmp_path: Path) -> None:
    from polylogue.cli.commands.embed import _record_archive_backfill_run

    _record_archive_backfill_run(
        tmp_path / "index.db",
        started_at_ms=1_767_225_700_000,
        status="stopped",
        processed_sessions=2,
        embedded_sessions=1,
        skipped_sessions=0,
        error_count=0,
        embedded_messages=3,
        estimated_cost_usd=0.001,
        stop_reason="time limit reached",
        configured_root=tmp_path,
    )

    with sqlite3.connect(tmp_path / "ops.db") as conn:
        row = conn.execute("SELECT status FROM embedding_catchup_runs WHERE run_id IS NOT NULL").fetchone()
    assert row == ("interrupted",)


def test_status_json_distinguishes_latest_material_archive_catchup(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="material-run",
            status="completed",
            started_at_ms=1_767_225_700_000,
            finished_at_ms=1_767_225_705_000,
            scanned_sessions=63,
            embedded_sessions=63,
            error_count=0,
            embedded_messages=2_818,
            estimated_cost_usd=0.1409,
        )
        upsert_embedding_catchup_run(
            conn,
            run_id="zero-progress-run",
            status="completed",
            started_at_ms=1_767_225_800_000,
            finished_at_ms=1_767_225_801_000,
            scanned_sessions=25,
            embedded_sessions=0,
            error_count=0,
            embedded_messages=0,
            estimated_cost_usd=0.0,
        )

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    material = payload["latest_material_catchup_run"]
    assert latest["run_id"] == "zero-progress-run"
    assert latest["processed_sessions"] == 25
    assert latest["embedded_messages"] == 0
    assert material["run_id"] == "material-run"
    assert material["embedded_sessions"] == 63
    assert material["embedded_messages"] == 2_818
    assert material["estimated_cost_usd"] == 0.1409


def test_status_json_treats_skipped_archive_catchup_as_material(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="skipped-run",
            status="completed",
            started_at_ms=1_767_225_900_000,
            finished_at_ms=1_767_225_901_000,
            scanned_sessions=25,
            embedded_sessions=0,
            skipped_sessions=25,
            error_count=0,
            embedded_messages=0,
            estimated_cost_usd=0.0,
        )

    payload = _run_status(db_anchor, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    material = payload["latest_material_catchup_run"]
    assert latest["run_id"] == "skipped-run"
    assert latest["skipped_sessions"] == 25
    assert material == latest


def test_status_json_reads_index_when_db_anchor_exists(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_without_embedding_ledgers(db_anchor, at_index_tier=False)
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(
        db_anchor,
        cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"),
    )

    assert payload["status"] == "partial"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1


def test_status_json_bypasses_schema_version_gate_for_operator_readiness(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 9")

    payload = _run_status(db_path, cfg=_cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert payload["status"] == "none"
    assert payload["config_enabled"] is False
    assert payload["has_voyage_api_key"] is True
    assert payload["configured_model"] == "voyage-4"
    assert payload["configured_dimension"] == 1024
    assert payload["monthly_cost_cap_usd"] == 5.0
    assert payload["pending_sessions"] == 2
    assert payload["next_action"] == {
        "code": "enable_embeddings",
        "command": "polylogue ops embed enable --yes",
        "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
    }


def test_status_json_counts_empty_vec0_rowids_as_zero_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path, vec_table=True)

    payload = _run_status(db_path)

    assert payload["embedded_messages"] == 0
    assert payload["retrieval_ready"] is False
    assert payload["freshness_status"] == "none"


@pytest.mark.parametrize(
    ("cfg", "config_enabled", "has_key", "stage_enabled"),
    [
        (_cfg(embedding_enabled=False, voyage_api_key="vk-live"), False, True, False),
        (_cfg(embedding_enabled=True, voyage_api_key=None), True, False, False),
    ],
)
def test_status_json_reports_config_gate_combinations(
    tmp_path: Path,
    cfg: PolylogueConfig,
    config_enabled: bool,
    has_key: bool,
    stage_enabled: bool,
) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path, cfg=cfg)

    assert payload["config_enabled"] is config_enabled
    assert payload["has_voyage_api_key"] is has_key
    assert payload["daemon_stage_enabled"] is stage_enabled
    assert payload["next_action"]["code"] == ("set_voyage_key" if not has_key else "enable_embeddings")


def test_status_json_detail_mode_stays_embedding_scoped(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path, "--detail")

    assert payload["pending_sessions"] == 2
    assert payload["pending_messages"] == 2
    assert payload["pending_messages_exact"] is True
    assert payload["retrieval_bands"] == {}


def test_status_json_detail_matches_archive_embedding_text_floor(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT
            );
            INSERT INTO sessions VALUES ('conv-1');
            INSERT INTO messages (
                message_id, session_id, text, role, message_type, material_origin, word_count, content_hash
            ) VALUES
                ('msg-long', 'conv-1', 'authored prose long enough', 'user', 'message', 'human_authored', 4, 'h1'),
                ('msg-short', 'conv-1', 'tiny', 'user', 'message', 'human_authored', 1, 'h2');
            """
        )
        conn.commit()

    payload = _run_status(index_db, "--detail")

    assert payload["pending_messages"] == 1
    assert payload["pending_messages_exact"] is True
    assert payload["candidate_prose_messages"] == 2
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["message_coverage_percent"] == 0.0


def test_status_json_includes_latest_catchup_run(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    # ``embedding_catchup_runs`` lives in the OPS tier, whose sole writer is
    # ops_write.upsert_embedding_catchup_run; the reader takes it from the
    # attached ops schema (status_payload.py:1416-1443). Seed the production
    # tier rather than the pre-split monolith shape.
    run_id = "legacy-run-1"
    with sqlite3.connect(_index_path(db_path).with_name("ops.db")) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_catchup_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                stop_reason TEXT,
                rebuild INTEGER NOT NULL DEFAULT 0,
                max_sessions INTEGER,
                max_messages INTEGER,
                stop_after_seconds INTEGER,
                max_errors INTEGER,
                planned_sessions INTEGER NOT NULL DEFAULT 0,
                planned_messages INTEGER NOT NULL DEFAULT 0,
                processed_sessions INTEGER NOT NULL DEFAULT 0,
                embedded_sessions INTEGER NOT NULL DEFAULT 0,
                skipped_sessions INTEGER NOT NULL DEFAULT 0,
                error_count INTEGER NOT NULL DEFAULT 0,
                embedded_messages INTEGER NOT NULL DEFAULT 0,
                estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
                last_session_id TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO embedding_catchup_runs (
                run_id, started_at, updated_at, completed_at, status, stop_reason,
                rebuild, max_sessions, max_messages, planned_sessions, planned_messages
            ) VALUES (?, datetime('now'), datetime('now'), datetime('now'), 'interrupted',
                      'keyboard interrupt', 1, 2, 10, 2, 2)
            """,
            (run_id,),
        )
        conn.commit()
    _stamp_tier(_index_path(db_path).with_name("ops.db"), "ops")

    payload = _run_status(db_path)

    latest = payload["latest_catchup_run"]
    assert latest["run_id"] == run_id
    assert latest["status"] == "interrupted"
    assert latest["stop_reason"] == "keyboard interrupt"
    assert latest["rebuild"] is True
    assert latest["planned_sessions"] == 2


def test_status_text_prints_machine_readable_next_action(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    output = _run_status_text(db_path, cfg=_cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert "Configured model:     voyage-4 (1024d)" in output
    assert "Monthly cost cap:     $5.00" in output
    assert "Total sessions:       2" in output
    assert "Embedded sessions:    0" in output
    assert "Embedded messages:    0" in output
    assert "Next action:          enable_embeddings" in output
    assert "Command:              polylogue ops embed enable --yes" in output


def test_status_text_prints_manual_backfill_when_config_disabled_but_partial(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    output = _run_status_text(db_anchor, cfg=_cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert "Next action:          continue_backfill" in output
    assert "Command:              polylogue ops embed backfill --yes --max-sessions 10" in output


def test_status_text_prints_daemon_catchup_when_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    output = _run_status_text(db_path, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert "Next action:          drain_backlog" in output
    assert "Command:              polylogue ops embed backfill --yes --max-sessions 10" in output


def test_status_json_reports_ready_next_action(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)
    # ``embedding_status``/``message_embeddings`` are read through the ATTACHED
    # embeddings schema (status_payload.py:991-998), never from ``main``, so
    # they belong in the embeddings tier file. Seeded into the index they were
    # invisible and the status read back ``none`` instead of ``complete``.
    with sqlite3.connect(_index_path(db_path).with_name("embeddings.db")) as conn:
        conn.execute(
            """
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                embedded_message_count INTEGER,
                needs_reindex INTEGER DEFAULT 0,
                error_message TEXT
            )
            """
        )
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO embedding_status (session_id, embedded_message_count, needs_reindex) VALUES (?, ?, 0)",
            [("conv-1", 1), ("conv-2", 1)],
        )
        conn.executemany("INSERT INTO message_embeddings (message_id) VALUES (?)", [("msg-1",), ("msg-2",)])
        conn.commit()
    _stamp_tier(_index_path(db_path).with_name("embeddings.db"), "embeddings")

    payload = _run_status(db_path, cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "complete"
    assert payload["retrieval_ready"] is True
    assert payload["next_action"] == {
        "code": "ready",
        "command": "polylogue --semantic <query>",
        "reason": "Embeddings are retrieval-ready.",
    }


def test_status_json_reports_metadata_only_row_as_not_ready(tmp_path: Path) -> None:
    """Embedding metadata alone must never certify a vector.

    The fixture publishes ``codex-session:complete`` through the production
    write route, so it starts out counted as embedded. This test then deletes
    *only* the physical vec0 vector, leaving ``message_embeddings_meta``,
    ``message_embedding_refs`` (with the current ``message_content_hash``),
    ``embedding_derivation_state`` ('succeeded') and ``embedding_status``
    exactly as production wrote them.

    Anti-vacuity: drop the
    ``EXISTS (SELECT 1 FROM message_embeddings ...)`` conjunct from
    ``_authoritative_archive_embedding_state`` and this test goes red -- the session
    would be reported embedded and retrieval-ready on the strength of metadata
    that describes a vector which is no longer there.
    """
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    embeddings_db = index_db.with_name("embeddings.db")

    before = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))
    assert before["embedded_sessions"] == 1
    assert before["embedded_messages"] == 1

    conn = sqlite3.connect(embeddings_db)
    try:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        with conn:
            conn.execute(
                """
                DELETE FROM message_embeddings
                WHERE vector_derivation_hash = (
                    SELECT lower(hex(vector_derivation_hash)) FROM message_embedding_refs
                    WHERE message_id = 'codex-session:complete:m1'
                )
                """
            )
        # Every metadata row production wrote is still present.
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone() == (1,)
        assert conn.execute("SELECT COUNT(*) FROM message_embedding_refs").fetchone() == (1,)
        assert conn.execute(
            "SELECT attempt_state FROM embedding_derivation_state WHERE session_id = 'codex-session:complete'"
        ).fetchone() == ("succeeded",)
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings").fetchone() == (0,)
    finally:
        conn.close()

    after = _run_status(db_anchor, "--detail", cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert after["embedded_sessions"] == 0
    assert after["embedded_messages"] == 0
    assert after["pending_sessions"] == 2
    assert after["retrieval_ready"] is False
    assert after["status"] != "complete"


def test_status_json_reports_terminal_failures_before_ready(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(db_anchor)
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(embeddings_db) as conn:
        # Go through the real attempt/failure write path (rather than a
        # bare embedding_status insert) so embedding_derivation_state also
        # carries the matching 'failed_terminal' row the modern blocked-
        # session predicate reads. embedding_failures comes from the
        # production EMBEDDINGS_DDL the fixture applies.
        attempt = begin_embedding_attempt(
            conn,
            session_id="codex-session:pending",
            origin="codex-session",
            source_hash=_pending_session_source_hash(),
            recipe=EmbeddingRecipe.current(model="voyage-4", dimensions=1024),
            started_at_ms=1_800_000_000_000,
        )
        record_embedding_failure(
            conn,
            session_id="codex-session:pending",
            origin="codex-session",
            message_refs=("codex-session:pending:m1",),
            provider="voyage",
            model="voyage-4",
            error_class="provider_http_400",
            error_message="Embedding generation failed: HTTP 400",
            retryable=False,
            occurred_at_ms=1_800_000_000_000,
            attempt=attempt,
        )
        conn.commit()

    payload = _run_status(
        db_anchor,
        "--detail",
        cfg=_cfg(embedding_enabled=True, voyage_api_key="vk-live"),
    )

    assert payload["status"] == "partial"
    assert payload["pending_sessions"] == 0
    assert payload["failure_count"] == 1
    assert payload["next_action"] == {
        "code": "inspect_failures",
        "command": "polylogue ops embed status --detail",
        "reason": "Embedding failures exist and need inspection before treating coverage as clean.",
    }
