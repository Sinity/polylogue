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
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Origin, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.derivation_identity import (
    DerivationIdentity,
    DerivationKey,
    DerivationKeyLike,
    DerivationSubject,
)
from polylogue.storage.embeddings.identity import EmbeddingRecipe, EmbeddingSourceDigest, vector_derivation_hash
from polylogue.storage.embeddings.materialization import (
    embed_archive_session_sync,
    select_pending_archive_session_window,
)
from polylogue.storage.embeddings.preflight import build_preflight_report
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.embedding_write import (
    begin_embedding_attempt,
    mark_session_embedding_error,
    resolve_embedding_failure,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
from tests.infra.live_ingest import write_index_session

T = TypeVar("T")

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


@pytest.fixture(autouse=True)
def _materialization_uses_freshness_baseline_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep initial materialization on the test's explicit voyage-4 baseline.

    Individual tests replace this patch with voyage-5 to prove a recipe change
    invalidates stale vectors before it can be published as current.
    """
    from polylogue.storage.embeddings import materialization

    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _EmbeddingConfig())


def _write_archive_session(
    root: Path,
    *,
    native_id: str,
    text: str,
    role: Role = Role.USER,
    material_origin: MaterialOrigin = MaterialOrigin.HUMAN_AUTHORED,
) -> str:
    with ArchiveStore(root) as archive:
        return write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=native_id,
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=role,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                        material_origin=material_origin,
                    )
                ],
            ),
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


def test_message_derivation_inspection_rejects_ref_after_message_semantics_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common-kernel route requires the exact current message semantics.

    Anti-vacuity: remove ``refs.message_content_hash = messages.content_hash``
    from adapter inspection and this stale ref is marked valid because its
    provider text, vector address, recipe, and output contract all match.
    """

    from polylogue.operations.embedding_derivation import make_embedding_frame
    from polylogue.storage.embeddings import materialization
    from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter

    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="semantic-identity", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"
    # The provider sees the same prose, but canonical message semantics include
    # author role/material origin and therefore have a different content hash.
    assert (
        _write_archive_session(
            root,
            native_id="semantic-identity",
            text=_INITIAL_TEXT,
            role=Role.ASSISTANT,
            material_origin=MaterialOrigin.ASSISTANT_AUTHORED,
        )
        == session_id
    )
    index_db = root / "index.db"
    monkeypatch.setattr(materialization, "load_polylogue_config", lambda: _EmbeddingConfig())
    adapter = EmbeddingDerivationAdapter(index_db, _FakeVectorProvider(), archive_root=root)
    frame = make_embedding_frame(index_db, archive_root=root, adapter=adapter, scope=(session_id,))
    keys, cursor = adapter.required_page(frame, cursor=None, limit=10)
    empty_scope = make_embedding_frame(index_db, archive_root=root, adapter=adapter, scope=())

    assert cursor is None
    assert keys
    assert adapter.inspect(frame, keys) == dict.fromkeys(keys, "stale")
    # A watcher path that resolves to no sessions must not turn into a full
    # archive sweep merely because an empty tuple is falsey.
    assert adapter.required_page(empty_scope, cursor=None, limit=10) == ((), None)
    assert adapter.excess_page(empty_scope, cursor=None, limit=10) == ((), None)
    quiet_adapter = EmbeddingDerivationAdapter(
        index_db,
        _FakeVectorProvider(),
        archive_root=root,
        quiet=lambda _frame, _key: True,
    )
    assert quiet_adapter.quiet(frame, keys[0]) is True


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


def test_status_payload_uses_its_resolved_recipe_for_exact_archive_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator status route must not reload a different ambient recipe."""
    from polylogue import config as config_module
    from polylogue.storage.embeddings.status_payload import embedding_status_payload

    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="status-recipe", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"

    monkeypatch.setattr(
        config_module,
        "load_polylogue_config",
        lambda: config_module.PolylogueConfig(_EmbeddingConfig(model="voyage-5")),
    )
    payload = embedding_status_payload(SimpleNamespace(config=SimpleNamespace(db_path=root / "index.db")))

    assert payload["configured_model"] == "voyage-5"
    assert payload["embedded_sessions"] == 0
    assert payload["pending_sessions"] == 1


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
        source.update(b"y" * 32)
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


def _current_ref_and_vector(conn: sqlite3.Connection, session_id: str) -> tuple[str, bytes, bytes]:
    """Resolve one session's (message_id, vector_derivation_hash, vector) via the
    v4 ref -> content-addressed row join (message_embeddings/
    message_embeddings_meta are keyed by vector_derivation_hash, not message_id,
    so message_embedding_refs is the required message_id -> hash bridge)."""
    row = conn.execute(
        """
        SELECT r.message_id, r.vector_derivation_hash, m.embedding
        FROM message_embedding_refs AS r
        JOIN message_embeddings AS m ON m.vector_derivation_hash = lower(hex(r.vector_derivation_hash))
        WHERE r.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return str(row[0]), bytes(row[1]), bytes(row[2])


def test_same_id_full_replace_reselects_and_atomically_replaces_vector_and_meta(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="full-replace", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(0.01), session_id).status == "embedded"

    conn = _open_embeddings(root / "embeddings.db")
    try:
        old_message_id, old_input_hash, old_vector = _current_ref_and_vector(conn, session_id)
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
        current_message_id = index_conn.execute(
            "SELECT message_id FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        new_message_id, new_input_hash, new_vector = _current_ref_and_vector(embed_conn, session_id)
    finally:
        index_conn.close()
        embed_conn.close()

    # message_id is stable across the full-replace (same native_id/position);
    # vector_derivation_hash -- identity-free, H(model, embedder input text) --
    # is what actually changes when the text changes, re-deriving to exactly
    # what the production identity function computes for the new text. The
    # vector itself is atomically replaced too (never left pointing at the
    # stale text's embedding).
    assert new_message_id == old_message_id == current_message_id
    assert new_input_hash == vector_derivation_hash(model="voyage-4", input_text=_CHANGED_TEXT)
    assert old_input_hash == vector_derivation_hash(model="voyage-4", input_text=_INITIAL_TEXT)
    assert new_input_hash != old_input_hash
    assert new_vector != old_vector


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


@pytest.mark.parametrize("damage", ["recipe", "missing-vector"])
def test_common_derivation_replaces_physical_vector_with_existing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    """Anti-vacuity: metadata-only publication leaves V1 or no vector, never V2."""
    from polylogue.daemon.derivation import DerivationRegistry, converge
    from polylogue.operations.embedding_derivation import make_embedding_frame
    from polylogue.storage.embeddings.derivation import EmbeddingDerivationAdapter
    from polylogue.storage.sqlite.write_lease import write_lease

    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="recipe-output", text=_INITIAL_TEXT)
    index_db = root / "index.db"
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(index_db, _FakeVectorProvider(0.01), session_id).status == "embedded"
    conn = _open_embeddings(root / "embeddings.db")
    try:
        _message_id, address, old_vector = _current_ref_and_vector(conn, session_id)
        if damage == "missing-vector":
            conn.execute("DELETE FROM message_embeddings WHERE vector_derivation_hash = ?", (address.hex(),))
            conn.commit()
    finally:
        conn.close()
    if damage == "recipe":
        monkeypatch.setattr("polylogue.storage.embeddings.identity.EMBEDDING_RECORD_SELECTOR", "changed-selector")

    def admit(actor: str, function: Callable[[], T]) -> T:
        with write_lease(actor, archive_root=root):
            return function()

    adapter = EmbeddingDerivationAdapter(index_db, _FakeVectorProvider(0.25), archive_root=root, reserve=admit)
    frame = make_embedding_frame(index_db, archive_root=root, adapter=adapter, scope=(session_id,))
    report = converge(DerivationRegistry([adapter]), frame, publisher=admit)
    assert report.done == 1, report.outcomes
    assert report.failed == 0
    conn = _open_embeddings(root / "embeddings.db")
    try:
        _, new_address, new_vector = _current_ref_and_vector(conn, session_id)
        assert new_address == address
        assert new_vector != old_vector
        import struct

        assert struct.unpack("<f", new_vector[:4])[0] == 0.25
    finally:
        conn.close()
    assert converge(DerivationRegistry([adapter]), frame).wrote_nothing


def test_unloadable_sqlite_vec_reports_unknown_coverage_not_a_measured_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete vector set must never be reported as ``none`` when unmeasurable.

    ``embeddings.db`` is the expensive-to-rebuild tier. When the sqlite-vec
    extension cannot load, readiness cannot be inspected at all -- reporting
    zero embedded sessions prescribes a paid regeneration of vectors that are
    present and intact.

    Anti-vacuity: restoring the former ``return None`` -> ``embedded_sessions
    = 0, pending_sessions = total`` collapse in
    ``_archive_embedding_status_payload`` makes this red -- ``status`` returns
    to ``"none"``, ``embedded_sessions`` to ``0``, and ``next_action`` back to
    the ``drain_backlog`` paid-backfill prescription.
    """
    from polylogue import config as config_module
    from polylogue.storage.embeddings import status_payload as status_payload_module
    from polylogue.storage.embeddings.status_payload import embedding_status_payload

    root = tmp_path / "archive"
    session_id = _write_archive_session(root, native_id="unmeasurable", text=_INITIAL_TEXT)
    initialize_archive_database(root / "embeddings.db", ArchiveTier.EMBEDDINGS)
    assert embed_archive_session_sync(root / "index.db", _FakeVectorProvider(), session_id).status == "embedded"

    monkeypatch.setattr(
        config_module,
        "load_polylogue_config",
        lambda: config_module.PolylogueConfig(_EmbeddingConfig()),
    )
    app = SimpleNamespace(config=SimpleNamespace(db_path=root / "index.db"))

    measured = embedding_status_payload(app)
    assert measured is not None
    assert measured["coverage_measurable"] is True
    assert measured["embedded_sessions"] == 1

    import polylogue.storage.sqlite.sqlite_vec_extension as vec_extension

    monkeypatch.setattr(
        vec_extension,
        "try_load_sqlite_vec",
        lambda conn: (False, ImportError("sqlite_vec is not installed")),
    )
    unmeasurable = embedding_status_payload(app)

    assert unmeasurable is not None
    assert unmeasurable["status"] == "unknown"
    assert unmeasurable["coverage_measurable"] is False
    assert "sqlite_vec_unavailable" in (unmeasurable["coverage_unmeasurable_reason"] or "")
    # The three states stay distinct: this is not a measured zero.
    assert unmeasurable["embedded_sessions"] is None
    assert unmeasurable["pending_sessions"] is None
    assert unmeasurable["embedded_messages"] is None
    assert unmeasurable["embedding_coverage_percent"] is None
    assert unmeasurable["retrieval_ready"] is False
    assert unmeasurable["next_action"]["code"] == "coverage_unmeasurable"
    assert status_payload_module.ArchiveEmbeddingStateProbe(counts=None, tier_absent=True).measurable is True


def test_absent_embeddings_tier_stays_a_measured_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The third state must not swallow the genuine one.

    Anti-vacuity: marking the tier-absent probe unmeasurable makes this red --
    an archive that truly has no vectors would stop reporting ``none`` and
    stop prescribing the backfill it genuinely needs.
    """
    from polylogue import config as config_module
    from polylogue.storage.embeddings.status_payload import embedding_status_payload

    root = tmp_path / "archive"
    _write_archive_session(root, native_id="no-vectors", text=_INITIAL_TEXT)
    (root / "embeddings.db").unlink(missing_ok=True)

    monkeypatch.setattr(
        config_module,
        "load_polylogue_config",
        lambda: config_module.PolylogueConfig(_EmbeddingConfig()),
    )
    payload = embedding_status_payload(SimpleNamespace(config=SimpleNamespace(db_path=root / "index.db")))

    assert payload is not None
    assert payload["coverage_measurable"] is True
    assert payload["embedded_sessions"] == 0
    assert payload["status"] == "none"
