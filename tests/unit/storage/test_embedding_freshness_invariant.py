"""Production-route proof for the monotonic embedding freshness invariant.

Each selector test starts with a genuinely materialized archive session, then
performs a same-id/same-count full replacement whose message content hash
changes while the compatibility ``embedding_status`` row remains clean.  The
only reason the session is selected is the shared exact DerivationKey
predicate.  The mutation named in each test is the historical bypass that made
that route silently trust the clean status row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.archive.message.roles import Role
from polylogue.cli.shared.types import AppEnv
from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derivation_identity import (
    DerivationIdentity,
    DerivationKey,
    DerivationKeyLike,
    DerivationSubject,
)
from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingSourceDigest
from polylogue.storage.embeddings.materialization import (
    EmbedSessionOutcome,
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.embeddings.preflight import PreflightReport, build_preflight_report
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    begin_embedding_attempt,
    mark_session_embedding_error,
    resolve_embedding_failure,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

_INITIAL_TEXT = "Initial authored archive prose that is long enough for an embedding vector."
_CHANGED_TEXT = "Changed authored archive prose that keeps the same identity and message count."


class _FakeVectorProvider:
    model = "voyage-4"
    dimension = 1024

    def __init__(self, value: float = 0.01) -> None:
        self.value = value

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "document"
        return [[self.value] * self.dimension for _ in texts]

    def upsert(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("archive materialization must use the archive embedding route")

    def query(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []

    def query_by_session(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        return []


class _EmbeddingConfig(dict[str, object]):
    embedding_model = "voyage-4"
    embedding_dimension = 1024
    embedding_max_cost_usd = 0.0

    def __init__(self, *, model: str = "voyage-4", api_key: str | None = "test-key") -> None:
        super().__init__(
            voyage_api_key=api_key,
            embedding_max_cost_usd=0.0,
            embedding_model=model,
            embedding_dimension=1024,
        )
        self.embedding_model = model
        self.embedding_dimension = 1024
        self.embedding_max_cost_usd = 0.0


def _write_archive_session(root: Path, *, native_id: str, text: str) -> str:
    with ArchiveStore(root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=MaterialOrigin.HUMAN_AUTHORED,
                    )
                ],
            )
        )


def _fresh_then_change(root: Path, *, native_id: str = "freshness-route") -> tuple[Path, Path, str]:
    session_id = _write_archive_session(root, native_id=native_id, text=_INITIAL_TEXT)
    index_db = root / "index.db"
    embeddings_db = root / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    outcome = embed_archive_session_sync(index_db, _FakeVectorProvider(), session_id)
    assert outcome.status == "embedded"

    replaced_session_id = _write_archive_session(root, native_id=native_id, text=_CHANGED_TEXT)
    assert replaced_session_id == session_id
    return index_db, embeddings_db, session_id


def _recipe(model: str = "voyage-4") -> EmbeddingRecipe:
    return EmbeddingRecipe.current(model=model, dimensions=1024)


def _open_embeddings(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        conn.close()
        pytest.skip(str(error) if error else "sqlite-vec extension is unavailable")
    return conn


def test_per_source_convergence_mutation_removing_shared_predicate_misses_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: convergence's per-source pending-id selector."""
    from polylogue.daemon import convergence_stages
    from polylogue.storage.embeddings import materialization

    index_db, _embeddings_db, session_id = _fresh_then_change(tmp_path / "archive")
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _EmbeddingConfig())

    with sqlite3.connect(index_db) as conn:
        selected = convergence_stages._archive_pending_embedding_session_ids(conn, [session_id])

    assert selected == [session_id]


def test_daemon_backlog_mutation_restoring_stale_check_bypass_misses_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: the daemon catch-up loop's real selection route."""
    from polylogue.daemon import embedding_backlog
    from polylogue.storage import search_providers
    from polylogue.storage.embeddings import materialization

    index_db, _embeddings_db, session_id = _fresh_then_change(tmp_path / "archive")
    selected: list[str] = []

    def _observe_embed(_index_db: Path, _provider: object, selected_session_id: str) -> EmbedSessionOutcome:
        selected.append(selected_session_id)
        return EmbedSessionOutcome(status="embedded", session_id=selected_session_id, embedded_message_count=1)

    monkeypatch.setattr(embedding_backlog, "load_polylogue_config", lambda: _EmbeddingConfig())
    monkeypatch.setattr(search_providers, "create_vector_provider", lambda **_kwargs: object())
    monkeypatch.setattr(materialization, "embed_archive_session_sync", _observe_embed)
    monkeypatch.setattr(embedding_backlog, "_upsert_archive_embedding_catchup_run", lambda *_args, **_kwargs: "run")

    assert embedding_backlog._drain_archive_embedding_backlog_once(index_db) == 1
    assert selected == [session_id]


def test_manual_backfill_mutation_restoring_stale_check_bypass_misses_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: ``polylogue embed`` archive backfill selection."""
    from polylogue.cli.commands import embed as embed_command
    from polylogue.storage.embeddings import materialization

    index_db, _embeddings_db, session_id = _fresh_then_change(tmp_path / "archive")
    selected: list[str] = []

    def _observe_embed(_index_db: Path, _provider: object, selected_session_id: str) -> EmbedSessionOutcome:
        selected.append(selected_session_id)
        return EmbedSessionOutcome(status="embedded", session_id=selected_session_id, embedded_message_count=1)

    monkeypatch.setattr(materialization, "embed_archive_session_sync", _observe_embed)
    monkeypatch.setattr(embed_command, "_record_archive_backfill_run", lambda *_args, **_kwargs: None)
    report = PreflightReport(
        total_sessions=1,
        pending_sessions=1,
        pending_messages=1,
        estimated_tokens=1,
        estimated_cost_usd=0.0,
        model="voyage-4",
        dimension=1024,
        cost_cap_usd=0.0,
    )
    env = cast(
        AppEnv, SimpleNamespace(ui=SimpleNamespace(console=SimpleNamespace(print=lambda *_args, **_kwargs: None)))
    )

    payload = embed_command._run_archive_backfill(
        env,
        index_db,
        object(),
        report,
        rebuild=False,
        max_sessions=None,
        stop_after_seconds=None,
        max_errors=None,
        output_format="json",
    )

    assert payload["candidate_sessions"] == 1
    assert selected == [session_id]


def test_preflight_mutation_restoring_stale_check_bypass_misses_changed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: the public preflight report, not a private SQL copy."""
    from polylogue import config as config_module

    index_db, _embeddings_db, _session_id = _fresh_then_change(tmp_path / "archive")
    monkeypatch.setattr(config_module, "load_polylogue_config", lambda: _EmbeddingConfig())

    report = build_preflight_report(index_db)

    assert report.total_sessions == 1
    assert report.pending_sessions == 1
    assert report.pending_messages == 1


def test_recipe_model_swap_makes_every_materialized_session_stale(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_ids = [
        _write_archive_session(root, native_id=native_id, text=f"{_INITIAL_TEXT} {native_id}")
        for native_id in ("recipe-a", "recipe-b")
    ]
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    for session_id in session_ids:
        assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"

    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(root / "embeddings.db"),))
        assert (
            select_pending_archive_session_window(
                conn,
                status_table="embeddings.embedding_status",
                recipe=_recipe("voyage-4"),
            )
            == []
        )
        swapped = select_pending_archive_session_window(
            conn,
            status_table="embeddings.embedding_status",
            recipe=_recipe("voyage-5"),
        )

    assert {item.session_id for item in swapped} == set(session_ids)


def test_config_change_then_old_terminal_error_cannot_clear_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deterministic actor ordering; deleting the generation/key WHERE guard fails."""
    from polylogue.daemon import convergence_stages

    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    conn = sqlite3.connect(embeddings_db)
    conn.row_factory = sqlite3.Row
    try:
        source = EmbeddingSourceDigest()
        source.update("codex-session:race:m1", b"x" * 32)
        old_attempt = begin_embedding_attempt(
            conn,
            session_id="codex-session:race",
            origin=Origin.CODEX_SESSION,
            source_hash=source.digest(),
            recipe=_recipe("voyage-4"),
            started_at_ms=1_800_000_000_000,
        )

        monkeypatch.setattr(
            convergence_stages,
            "load_polylogue_config",
            lambda: _EmbeddingConfig(model="voyage-5"),
        )
        convergence_stages._reconcile_embedding_config_change(conn)
        conn.commit()

        status = mark_session_embedding_error(
            conn,
            session_id=old_attempt.session_id,
            origin=Origin.CODEX_SESSION,
            error_message="Embedding generation failed: HTTP 400",
            retryable=False,
            attempt=old_attempt,
        )
        state = conn.execute(
            """
            SELECT generation, recipe_hash, attempt_state
            FROM embedding_derivation_state WHERE session_id = ?
            """,
            (old_attempt.session_id,),
        ).fetchone()
        failure = conn.execute(
            """
            SELECT lifecycle_state, generation, derivation_key
            FROM embedding_failures WHERE session_id = ? ORDER BY created_at_ms DESC LIMIT 1
            """,
            (old_attempt.session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert state is not None
    assert state["generation"] == old_attempt.generation + 1
    assert bytes(state["recipe_hash"]) == _recipe("voyage-5").recipe_hash
    assert state["attempt_state"] == "pending"
    assert status.needs_reindex is True
    assert status.error_message is None
    assert failure is not None
    assert failure["lifecycle_state"] == "superseded"
    assert failure["generation"] == old_attempt.generation
    assert bytes(failure["derivation_key"]) == old_attempt.derivation_key


def test_archive_check_reconciles_recipe_on_sibling_embeddings_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: convergence must advance ``embeddings.db``, not ``index.db``."""
    from polylogue.daemon import convergence_stages
    from polylogue.storage.embeddings import materialization

    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="sibling-recipe", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"

    with sqlite3.connect(root / "embeddings.db") as conn:
        before = conn.execute(
            "SELECT generation FROM embedding_derivation_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    assert before is not None

    changed_config = _EmbeddingConfig(model="voyage-5")
    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: changed_config)
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: changed_config)

    assert convergence_stages._archive_embed_check_sessions(root / "index.db", [session_id]) == {session_id}

    with sqlite3.connect(root / "embeddings.db") as conn:
        state = conn.execute(
            """
            SELECT generation, recipe_hash, attempt_state
            FROM embedding_derivation_state WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        status = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    assert state is not None
    assert state[0] == int(before[0]) + 1
    assert bytes(state[1]) == _recipe("voyage-5").recipe_hash
    assert state[2] == "pending"
    assert status == (1, None)


def test_recipe_change_reconciliation_does_not_restarve_sessions_already_succeeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: convergence must not un-fresh already-succeeded sessions.

    When an archive has more pending sessions than one bounded convergence
    window (``_DAEMON_EMBED_MAX_SESSIONS``), ``message_embeddings_meta`` for
    the not-yet-reembedded remainder still carries the old recipe hash on the
    next convergence pass, so ``meta_recipe_changed`` stays true archive-wide.
    The bulk ``embedding_status`` reindex mark must scope to the sessions
    whose ``embedding_derivation_state`` generation actually advanced *this*
    pass, not the whole table -- otherwise sessions that already succeeded
    under the new generation/key get re-flagged ``needs_reindex=1`` on every
    subsequent pass and the daemon loops on the same first batch forever
    instead of covering the rest of the archive (polylogue PR #3067 review).
    """
    from polylogue.daemon import convergence_stages
    from polylogue.storage.embeddings import materialization

    root = tmp_path / "archive"
    session_ids = [
        _write_archive_session(root, native_id=f"progress-{index:02d}", text=f"{_INITIAL_TEXT} {index}")
        for index in range(30)
    ]
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    for session_id in session_ids:
        assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"

    changed_config = _EmbeddingConfig(model="voyage-5")
    monkeypatch.setattr(convergence_stages, "load_polylogue_config", lambda: changed_config)
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: changed_config)

    # Pass 1: recipe change is reconciled. Every session becomes pending, but
    # the production check function only ever returns one bounded window.
    first_batch = convergence_stages._archive_embed_check_sessions(root / "index.db", session_ids)
    assert len(first_batch) == convergence_stages._DAEMON_EMBED_MAX_SESSIONS
    assert first_batch.issubset(set(session_ids))
    remaining = set(session_ids) - first_batch
    assert remaining

    # The daemon actually re-embeds exactly the returned first batch under
    # the new recipe -- the rest of the archive is still untouched.
    new_recipe_provider = _FakeVectorProvider(0.02)
    new_recipe_provider.model = "voyage-5"
    for session_id in first_batch:
        outcome = embed_archive_session_sync(root / "index.db", new_recipe_provider, session_id)
        assert outcome.status == "embedded"

    with sqlite3.connect(root / "embeddings.db") as conn:
        needs_reindex_after_batch = dict(
            conn.execute("SELECT session_id, needs_reindex FROM embedding_status").fetchall()
        )
    assert all(needs_reindex_after_batch[sid] == 0 for sid in first_batch)

    # Pass 2: convergence runs again before the remaining sessions are
    # embedded. meta_recipe_changed is still true (the remaining sessions'
    # message_embeddings_meta rows still carry the old recipe hash), so the
    # bug reproduces here if the bulk mark is not scoped: it would re-flag the
    # already-succeeded first batch as needs_reindex=1 and the returned
    # pending set would loop back over the same first batch instead of
    # advancing to `remaining`.
    second_batch = convergence_stages._archive_embed_check_sessions(root / "index.db", session_ids)

    with sqlite3.connect(root / "embeddings.db") as conn:
        needs_reindex_after_pass2 = dict(
            conn.execute("SELECT session_id, needs_reindex FROM embedding_status").fetchall()
        )

    assert all(needs_reindex_after_pass2[sid] == 0 for sid in first_batch), (
        "sessions that already succeeded under the new recipe/generation must not be re-marked needs_reindex=1"
    )
    assert second_batch == remaining, "convergence must advance to the untouched remainder, not loop on batch one"


def test_unscoped_or_legacy_failure_receipt_cannot_project_over_keyed_generation(tmp_path: Path) -> None:
    """Mutation: allowing generation-zero status writes clears a newer pending mark."""
    embeddings_db = tmp_path / "embeddings.db"
    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    conn = sqlite3.connect(embeddings_db)
    conn.row_factory = sqlite3.Row
    try:
        legacy_failure = mark_session_embedding_error(
            conn,
            session_id="codex-session:legacy-race",
            origin=Origin.CODEX_SESSION,
            error_message="old terminal failure",
            retryable=False,
        )
        failure_id = str(
            conn.execute(
                "SELECT failure_id FROM embedding_failures WHERE session_id = ?",
                (legacy_failure.session_id,),
            ).fetchone()[0]
        )

        source = EmbeddingSourceDigest()
        source.update("codex-session:legacy-race:m1", b"y" * 32)
        attempt = begin_embedding_attempt(
            conn,
            session_id=legacy_failure.session_id,
            origin=Origin.CODEX_SESSION,
            source_hash=source.digest(),
            recipe=_recipe(),
            started_at_ms=1_800_000_000_001,
        )

        resolve_embedding_failure(conn, failure_id=failure_id, action="acknowledge")
        status_after_legacy_resolution = conn.execute(
            "SELECT needs_reindex, error_message FROM embedding_status WHERE session_id = ?",
            (attempt.session_id,),
        ).fetchone()
        state_after_legacy_resolution = conn.execute(
            "SELECT generation, derivation_key, attempt_state FROM embedding_derivation_state WHERE session_id = ?",
            (attempt.session_id,),
        ).fetchone()

        unscoped = mark_session_embedding_error(
            conn,
            session_id=attempt.session_id,
            origin=Origin.CODEX_SESSION,
            error_message="unscoped late failure",
            retryable=False,
        )
        latest_receipt = conn.execute(
            """
            SELECT lifecycle_state, generation, derivation_key
            FROM embedding_failures
            WHERE session_id = ?
            ORDER BY created_at_ms DESC, failure_id DESC
            LIMIT 1
            """,
            (attempt.session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert tuple(status_after_legacy_resolution) == (1, None)
    assert state_after_legacy_resolution is not None
    assert state_after_legacy_resolution[0] == attempt.generation
    assert bytes(state_after_legacy_resolution[1]) == attempt.derivation_key
    assert state_after_legacy_resolution[2] == "pending"
    assert unscoped.needs_reindex is True
    assert unscoped.error_message is None
    assert latest_receipt is not None
    assert latest_receipt["lifecycle_state"] == "superseded"
    assert latest_receipt["generation"] == 0
    assert latest_receipt["derivation_key"] is None


def test_same_id_full_replace_reselects_and_atomically_replaces_vector_and_meta(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="full-replace", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(0.01), session_id).status == "embedded"

    conn = _open_embeddings(root / "embeddings.db")
    try:
        old_message_id, old_hash, old_vector = conn.execute(
            """
            SELECT m.message_id, mm.content_hash, m.embedding
            FROM message_embeddings AS m
            JOIN message_embeddings_meta AS mm USING (message_id)
            WHERE m.session_id = ?
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    assert _write_archive_session(root, native_id="full-replace", text=_CHANGED_TEXT) == session_id
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("ATTACH DATABASE ? AS embeddings", (str(root / "embeddings.db"),))
        pending = select_pending_archive_session_window(
            conn,
            status_table="embeddings.embedding_status",
            recipe=_recipe(),
        )
    assert [item.session_id for item in pending] == [session_id]

    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(0.02), session_id).status == "embedded"
    index_conn = sqlite3.connect(root / "index.db")
    embed_conn = _open_embeddings(root / "embeddings.db")
    try:
        current_message_id, current_hash = index_conn.execute(
            "SELECT message_id, content_hash FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        rows = embed_conn.execute(
            """
            SELECT m.message_id, mm.content_hash, m.embedding
            FROM message_embeddings AS m
            JOIN message_embeddings_meta AS mm USING (message_id)
            WHERE m.session_id = ?
            """,
            (session_id,),
        ).fetchall()
    finally:
        index_conn.close()
        embed_conn.close()

    assert len(rows) == 1
    assert rows[0][0] == old_message_id == current_message_id
    assert bytes(rows[0][1]) == bytes(current_hash)
    assert bytes(rows[0][1]) != bytes(old_hash)
    assert bytes(rows[0][2]) != bytes(old_vector)


def test_derivation_key_value_shape_is_storage_neutral_and_generation_free() -> None:
    key = DerivationKey(
        subject=DerivationSubject(reference="session:1", grain="message-vectors"),
        source_identity=DerivationIdentity.from_mapping("source.v1", {"sha256": b"s" * 32}),
        recipe_identity=DerivationIdentity.from_mapping("recipe.v1", {"model": "voyage-4"}),
        output_contract=DerivationIdentity.from_mapping("output.v1", {"dimensions": 1024}),
    )

    assert isinstance(key, DerivationKeyLike)
    assert tuple(field.name for field in fields(DerivationKey)) == (
        "subject",
        "source_identity",
        "recipe_identity",
        "output_contract",
    )
    assert key.digest() == key.digest()
    assert all(
        excluded not in key.canonical_bytes().decode("utf-8")
        for excluded in ("generation", "producer", "eligibility", "privacy", "result_hash")
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("canonicalization", "changed-canonicalization"),
        ("record_selector", "changed-selector"),
        ("chunking_version", "changed-chunking"),
        ("provider", "changed-provider"),
        ("model", "changed-model"),
        ("model_revision", "changed-revision"),
        ("dimensions", 768),
        ("task", "changed-task"),
        ("input_type", "changed-input-type"),
        ("normalization", "changed-normalization"),
        ("tool_implementation", "changed-tool"),
        ("input_schema_version", "changed-input-schema"),
    ],
)
def test_recipe_mutation_removing_any_declared_computational_field_preserves_wrong_reuse(
    field_name: str,
    changed_value: str | int,
) -> None:
    baseline = _recipe()
    # dataclasses.replace can't statically verify a **dict of heterogeneous
    # per-field values against EmbeddingRecipe's precise field types; the
    # parametrization above is the source of truth for valid field/value pairs.
    changed = replace(baseline, **{field_name: changed_value})  # type: ignore[arg-type]

    assert changed.recipe_hash != baseline.recipe_hash
