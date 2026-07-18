"""Archive embedding metadata write helpers.

Writer module: embeddings.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.identity import (
    EmbeddingRecipe,
    EmbeddingSourceDigest,
    embedding_derivation_key,
    message_embedding_derivation_digest_from_hashes,
    message_embedding_derivation_key,
)
from polylogue.storage.search_providers.sqlite_vec_support import _serialize_f32
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingStatus:
    session_id: str
    origin: str
    message_count_embedded: int
    last_embedded_at_ms: int | None
    needs_reindex: bool
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingMeta:
    message_id: str
    model: str
    dimension: int
    content_hash: bytes
    embedded_at_ms: int | None
    needs_reindex: bool


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingAttempt:
    """Captured key/generation that conditionally owns one terminal write."""

    session_id: str
    origin: str
    generation: int
    derivation_key: bytes
    source_hash: bytes
    recipe_hash: bytes
    output_contract_hash: bytes


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingWrite:
    message_id: str
    session_id: str
    origin: Origin | str
    embedding: list[float]
    model: str
    embedded_at_ms: int
    content_hash: bytes
    recipe_hash: bytes | None = None
    derivation_key: bytes | None = None
    generation: int = 0


EmbeddingFailureState = Literal["retryable", "terminal", "acknowledged", "superseded", "resolved"]
EmbeddingFailureResolution = Literal["acknowledge", "requeue", "supersede"]


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingFailure:
    failure_id: str
    session_id: str
    origin: str
    message_refs: tuple[str, ...]
    provider: str
    model: str
    error_class: str
    error_message: str
    retryable: bool
    lifecycle_state: EmbeddingFailureState
    created_at_ms: int
    updated_at_ms: int
    resolved_at_ms: int | None
    resolution_action: str | None
    resolution_note: str | None
    superseded_by: str | None
    generation: int = 0
    derivation_key: bytes | None = None
    source_hash: bytes | None = None
    recipe_hash: bytes | None = None


def begin_embedding_attempt(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    source_hash: bytes,
    recipe: EmbeddingRecipe,
    started_at_ms: int | None = None,
) -> ArchiveEmbeddingAttempt:
    """Capture the exact desired key and advance this session's generation."""

    if len(source_hash) != 32:
        raise ValueError("source_hash must be a SHA-256 value")
    now_ms = int(time.time() * 1000) if started_at_ms is None else started_at_ms
    origin_value = _enum_value(origin)
    key = embedding_derivation_key(session_id=session_id, source_hash=source_hash, recipe=recipe)
    derivation_key = key.digest()
    recipe_hash = recipe.recipe_hash
    output_contract_hash = recipe.output_contract_hash
    with conn:
        row = conn.execute(
            """
            INSERT INTO embedding_derivation_state (
                session_id, origin, generation, derivation_key, source_hash, recipe_hash,
                output_contract_hash, attempt_state, message_count, updated_at_ms
            ) VALUES (?, ?, 1, ?, ?, ?, ?, 'pending', 0, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                generation = embedding_derivation_state.generation + 1,
                derivation_key = excluded.derivation_key,
                source_hash = excluded.source_hash,
                recipe_hash = excluded.recipe_hash,
                output_contract_hash = excluded.output_contract_hash,
                attempt_state = 'pending',
                message_count = 0,
                updated_at_ms = excluded.updated_at_ms
            RETURNING generation
            """,
            (
                session_id,
                origin_value,
                derivation_key,
                source_hash,
                recipe_hash,
                output_contract_hash,
                now_ms,
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("embedding attempt generation was not returned")
        generation = int(row[0])
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, 0, NULL, 1, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                needs_reindex = 1,
                error_message = NULL
            """,
            (session_id, origin_value),
        )
    return ArchiveEmbeddingAttempt(
        session_id=session_id,
        origin=origin_value,
        generation=generation,
        derivation_key=derivation_key,
        source_hash=source_hash,
        recipe_hash=recipe_hash,
        output_contract_hash=output_contract_hash,
    )


def supersede_embedding_attempt(
    conn: sqlite3.Connection,
    *,
    attempt: ArchiveEmbeddingAttempt,
    source_hash: bytes,
    recipe: EmbeddingRecipe,
    updated_at_ms: int | None = None,
) -> ArchiveEmbeddingAttempt | None:
    """Advance an exact in-flight attempt to a newer desired key without publishing.

    The conditional generation transition is useful when source/config identity
    moves while a provider request is in flight. It cannot overwrite a generation
    that another reconciler or worker has already advanced.
    """

    if len(source_hash) != 32:
        raise ValueError("source_hash must be a SHA-256 value")
    now_ms = int(time.time() * 1000) if updated_at_ms is None else updated_at_ms
    derivation_key = embedding_derivation_key(
        session_id=attempt.session_id,
        source_hash=source_hash,
        recipe=recipe,
    ).digest()
    recipe_hash = recipe.recipe_hash
    output_contract_hash = recipe.output_contract_hash
    with conn:
        row = conn.execute(
            """
            UPDATE embedding_derivation_state
            SET generation = generation + 1,
                derivation_key = ?,
                source_hash = ?,
                recipe_hash = ?,
                output_contract_hash = ?,
                attempt_state = 'pending',
                message_count = 0,
                updated_at_ms = ?
            WHERE session_id = ?
              AND generation = ?
              AND derivation_key = ?
              AND attempt_state = 'pending'
            RETURNING generation
            """,
            (
                derivation_key,
                source_hash,
                recipe_hash,
                output_contract_hash,
                now_ms,
                attempt.session_id,
                attempt.generation,
                attempt.derivation_key,
            ),
        ).fetchone()
        if row is None:
            return None
        generation = int(row[0])
        conn.execute(
            """
            UPDATE embedding_status
            SET needs_reindex = 1, error_message = NULL
            WHERE session_id = ?
            """,
            (attempt.session_id,),
        )
    return ArchiveEmbeddingAttempt(
        session_id=attempt.session_id,
        origin=attempt.origin,
        generation=generation,
        derivation_key=derivation_key,
        source_hash=source_hash,
        recipe_hash=recipe_hash,
        output_contract_hash=output_contract_hash,
    )


def upsert_message_embedding(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    origin: Origin | str,
    embedding: list[float],
    model: str,
    embedded_at_ms: int,
    content_hash: bytes | None = None,
) -> ArchiveEmbeddingMeta:
    """Upsert one message vector plus its reproducibility metadata."""
    if content_hash is None:
        raise ValueError("content_hash is required for message embedding metadata")
    upsert_message_embeddings(
        conn,
        [
            ArchiveEmbeddingWrite(
                message_id=message_id,
                session_id=session_id,
                origin=origin,
                embedding=embedding,
                model=model,
                embedded_at_ms=embedded_at_ms,
                content_hash=content_hash,
            )
        ],
    )
    return read_embedding_meta(conn, message_id)


def _prepared_write(write: ArchiveEmbeddingWrite) -> ArchiveEmbeddingWrite:
    if len(write.embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"embedding must have {EMBEDDING_DIMENSION} dimensions")
    if len(write.content_hash) != 32:
        raise ValueError("content_hash must be a SHA-256 value")
    recipe = EmbeddingRecipe.current(model=write.model, dimensions=EMBEDDING_DIMENSION)
    recipe_hash = write.recipe_hash or recipe.recipe_hash
    derivation_key = (
        write.derivation_key
        or message_embedding_derivation_key(
            message_id=write.message_id,
            content_hash=write.content_hash,
            recipe=recipe,
        ).digest()
    )
    return ArchiveEmbeddingWrite(
        message_id=write.message_id,
        session_id=write.session_id,
        origin=write.origin,
        embedding=write.embedding,
        model=write.model,
        embedded_at_ms=write.embedded_at_ms,
        content_hash=write.content_hash,
        recipe_hash=recipe_hash,
        derivation_key=derivation_key,
        generation=write.generation,
    )


def _write_message_embeddings(conn: sqlite3.Connection, writes: Sequence[ArchiveEmbeddingWrite]) -> None:
    for raw_write in writes:
        write = _prepared_write(raw_write)
        origin_value = _enum_value(write.origin)
        conn.execute("DELETE FROM message_embeddings WHERE message_id = ?", (write.message_id,))
        conn.execute(
            """
            INSERT INTO message_embeddings (message_id, embedding, session_id, origin)
            VALUES (?, ?, ?, ?)
            """,
            (write.message_id, _serialize_f32(write.embedding), write.session_id, origin_value),
        )
        conn.execute(
            """
            INSERT INTO message_embeddings_meta (
                message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex,
                recipe_hash, derivation_key, generation
            ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                model = excluded.model,
                dimension = excluded.dimension,
                content_hash = excluded.content_hash,
                embedded_at_ms = excluded.embedded_at_ms,
                needs_reindex = 0,
                recipe_hash = excluded.recipe_hash,
                derivation_key = excluded.derivation_key,
                generation = excluded.generation
            """,
            (
                write.message_id,
                write.model,
                EMBEDDING_DIMENSION,
                write.content_hash,
                write.embedded_at_ms,
                write.recipe_hash,
                write.derivation_key,
                write.generation,
            ),
        )


def upsert_message_embeddings(
    conn: sqlite3.Connection,
    writes: Sequence[ArchiveEmbeddingWrite],
) -> None:
    """Upsert message vectors and metadata in one transaction.

    Session materialization should use :func:`complete_embedding_attempt_success`
    so vector/meta replacement and the generation-guarded terminal state commit
    atomically.  This lower-level helper remains for direct one-off writers.
    """

    with conn:
        _write_message_embeddings(conn, writes)


def complete_embedding_attempt_success(
    conn: sqlite3.Connection,
    *,
    attempt: ArchiveEmbeddingAttempt,
    writes: Sequence[ArchiveEmbeddingWrite],
    completed_at_ms: int | None = None,
) -> bool:
    """Atomically replace one session's vectors and clear debt for this exact attempt.

    Returns ``False`` when a newer generation/key has superseded the caller;
    no vector, metadata, status, or failure lifecycle rows are then changed.
    """

    now_ms = int(time.time() * 1000) if completed_at_ms is None else completed_at_ms
    prepared = tuple(_prepared_write(write) for write in writes)
    message_ids: set[str] = set()
    source_digest = EmbeddingSourceDigest()
    for write in sorted(prepared, key=lambda item: item.message_id):
        if write.session_id != attempt.session_id:
            raise ValueError("all embedding writes must belong to the attempt session")
        if write.message_id in message_ids:
            raise ValueError("embedding attempt writes must have unique message ids")
        message_ids.add(write.message_id)
        if write.generation != attempt.generation:
            raise ValueError("embedding write generation must match the owning attempt")
        if write.recipe_hash != attempt.recipe_hash:
            raise ValueError("embedding write recipe must match the owning attempt")
        expected_message_key = message_embedding_derivation_digest_from_hashes(
            message_id=write.message_id,
            content_hash=write.content_hash,
            recipe_hash=attempt.recipe_hash,
            output_contract_hash=attempt.output_contract_hash,
        )
        if write.derivation_key != expected_message_key:
            raise ValueError("embedding write key must match its message content and attempt recipe")
        source_digest.update(write.message_id, write.content_hash)
    if source_digest.digest() != attempt.source_hash:
        raise ValueError("embedding writes must cover the attempt's exact source identity")
    message_count = len(prepared)

    with conn:
        current = conn.execute(
            """
            SELECT 1
            FROM embedding_derivation_state
            WHERE session_id = ?
              AND generation = ?
              AND derivation_key = ?
              AND attempt_state = 'pending'
            """,
            (attempt.session_id, attempt.generation, attempt.derivation_key),
        ).fetchone()
        if current is None:
            return False

        prior_message_ids = tuple(
            str(row[0])
            for row in conn.execute(
                "SELECT message_id FROM message_embeddings WHERE session_id = ?",
                (attempt.session_id,),
            ).fetchall()
        )
        conn.execute("DELETE FROM message_embeddings WHERE session_id = ?", (attempt.session_id,))
        if prior_message_ids:
            placeholders = ", ".join("?" for _ in prior_message_ids)
            conn.execute(
                f"DELETE FROM message_embeddings_meta WHERE message_id IN ({placeholders})",
                prior_message_ids,
            )
        _write_message_embeddings(conn, prepared)
        updated = conn.execute(
            """
            UPDATE embedding_derivation_state
            SET attempt_state = 'succeeded', message_count = ?, updated_at_ms = ?
            WHERE session_id = ?
              AND generation = ?
              AND derivation_key = ?
              AND attempt_state = 'pending'
            """,
            (message_count, now_ms, attempt.session_id, attempt.generation, attempt.derivation_key),
        )
        if updated.rowcount != 1:
            raise RuntimeError("embedding generation changed inside a guarded success transaction")
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, ?, ?, 0, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at_ms = excluded.last_embedded_at_ms,
                needs_reindex = 0,
                error_message = NULL
            """,
            (attempt.session_id, attempt.origin, message_count, now_ms),
        )
        conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = 'resolved', updated_at_ms = ?, resolved_at_ms = ?,
                resolution_action = 'embedded'
            WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
            """,
            (now_ms, now_ms, attempt.session_id),
        )
    return True


def mark_session_embedding_error(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    error_message: str,
    retryable: bool = True,
    attempt: ArchiveEmbeddingAttempt | None = None,
) -> ArchiveEmbeddingStatus:
    """Record a failure with a status projection for compatibility callers.

    A legacy database with no derivation row retains the old projection
    behavior. Once a derivation row exists, callers need its captured attempt
    token to mutate freshness state.
    """

    record_embedding_failure(
        conn,
        session_id=session_id,
        origin=origin,
        message_refs=(),
        provider="unknown",
        model="unknown",
        error_class="embedding_error",
        error_message=error_message,
        retryable=retryable,
        attempt=attempt,
    )
    return read_embedding_status(conn, session_id)


def record_embedding_failure(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    message_refs: Sequence[str],
    provider: str,
    model: str,
    error_class: str,
    error_message: str,
    retryable: bool,
    occurred_at_ms: int | None = None,
    attempt: ArchiveEmbeddingAttempt | None = None,
) -> ArchiveEmbeddingFailure:
    """Persist one inspectable failure and conditionally project its current state."""

    if attempt is not None and attempt.session_id != session_id:
        raise ValueError("embedding failure session must match the captured attempt")
    now_ms = int(time.time() * 1000) if occurred_at_ms is None else occurred_at_ms
    failure_id = f"embedding-failure:{uuid.uuid4()}"
    origin_value = _enum_value(origin)
    refs = tuple(dict.fromkeys(str(ref) for ref in message_refs))
    desired_state = "failed_retryable" if retryable else "failed_terminal"
    lifecycle: EmbeddingFailureState = "retryable" if retryable else "terminal"
    applied = False
    stale_attempt = False
    legacy_projection = False
    with conn:
        if attempt is not None:
            updated = conn.execute(
                """
                UPDATE embedding_derivation_state
                SET attempt_state = ?, updated_at_ms = ?
                WHERE session_id = ?
                  AND generation = ?
                  AND derivation_key = ?
                  AND attempt_state = 'pending'
                """,
                (desired_state, now_ms, session_id, attempt.generation, attempt.derivation_key),
            )
            applied = updated.rowcount == 1
            stale_attempt = not applied
        else:
            legacy_projection = not _session_has_embedding_derivation_state(conn, session_id)
            if not legacy_projection:
                # An unscoped receipt cannot be authoritative once this
                # session has an exact key/generation. Preserve it only as
                # superseded evidence; never project it onto current state.
                stale_attempt = True
                lifecycle = "superseded"
        if applied:
            conn.execute(
                """
                INSERT INTO embedding_status (
                    session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
                ) VALUES (?, ?, 0, NULL, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    origin = excluded.origin,
                    needs_reindex = excluded.needs_reindex,
                    error_message = excluded.error_message
                """,
                (session_id, origin_value, 1 if retryable else 0, error_message),
            )
            conn.execute(
                """
                UPDATE embedding_failures
                SET lifecycle_state = 'superseded', updated_at_ms = ?, resolved_at_ms = ?,
                    resolution_action = 'superseded', superseded_by = ?
                WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
                """,
                (now_ms, now_ms, failure_id, session_id),
            )
        elif stale_attempt:
            lifecycle = "superseded"
        elif legacy_projection:
            # Databases/callers predating the derivation ledger keep their
            # established failure lifecycle. This branch cannot clobber a
            # newer generation because no generation exists for the session.
            conn.execute(
                """
                INSERT INTO embedding_status (
                    session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
                ) VALUES (?, ?, 0, NULL, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    origin = excluded.origin,
                    needs_reindex = excluded.needs_reindex,
                    error_message = excluded.error_message
                """,
                (session_id, origin_value, 1 if retryable else 0, error_message),
            )
            conn.execute(
                """
                UPDATE embedding_failures
                SET lifecycle_state = 'superseded', updated_at_ms = ?, resolved_at_ms = ?,
                    resolution_action = 'superseded', superseded_by = ?
                WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
                """,
                (now_ms, now_ms, failure_id, session_id),
            )
        conn.execute(
            """
            INSERT INTO embedding_failures (
                failure_id, session_id, origin, message_refs_json, provider, model,
                error_class, error_message, retryable, lifecycle_state, created_at_ms, updated_at_ms,
                resolved_at_ms, resolution_action,
                generation, derivation_key, source_hash, recipe_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failure_id,
                session_id,
                origin_value,
                json.dumps(refs),
                provider,
                model,
                error_class,
                error_message,
                int(retryable),
                lifecycle,
                now_ms,
                now_ms,
                now_ms if stale_attempt else None,
                "superseded" if stale_attempt else None,
                0 if attempt is None else attempt.generation,
                None if attempt is None else attempt.derivation_key,
                None if attempt is None else attempt.source_hash,
                None if attempt is None else attempt.recipe_hash,
            ),
        )
    return read_embedding_failure(conn, failure_id)


def resolve_embedding_failure(
    conn: sqlite3.Connection,
    *,
    failure_id: str,
    action: EmbeddingFailureResolution,
    note: str | None = None,
    superseded_by: str | None = None,
    resolved_at_ms: int | None = None,
) -> ArchiveEmbeddingFailure:
    """Explicitly acknowledge, supersede, or requeue an active failure."""

    now_ms = int(time.time() * 1000) if resolved_at_ms is None else resolved_at_ms
    state = cast(
        EmbeddingFailureState,
        {
            "acknowledge": "acknowledged",
            "supersede": "superseded",
            "requeue": "resolved",
        }[action],
    )
    identity_select = _embedding_failure_identity_select(conn)
    with conn:
        row = conn.execute(
            f"""
            SELECT session_id, {identity_select}
            FROM embedding_failures
            WHERE failure_id = ? AND lifecycle_state IN ('retryable', 'terminal')
            """,
            (failure_id,),
        ).fetchone()
        if row is None:
            raise KeyError(failure_id)
        session_id = str(row[0])
        generation = int(row[1])
        derivation_key = row[2]
        conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = ?, updated_at_ms = ?, resolved_at_ms = ?, resolution_action = ?,
                resolution_note = ?, superseded_by = ?
            WHERE failure_id = ?
            """,
            (state, now_ms, now_ms, action, note, superseded_by, failure_id),
        )
        if action == "requeue" and generation > 0 and derivation_key is not None:
            updated = conn.execute(
                """
                UPDATE embedding_derivation_state
                SET attempt_state = 'pending', updated_at_ms = ?
                WHERE session_id = ? AND generation = ? AND derivation_key = ?
                  AND attempt_state IN ('failed_retryable', 'failed_terminal')
                """,
                (now_ms, session_id, generation, derivation_key),
            )
            if updated.rowcount == 1:
                conn.execute(
                    "UPDATE embedding_status SET needs_reindex = 1, error_message = NULL WHERE session_id = ?",
                    (session_id,),
                )
        elif action != "requeue" and generation > 0 and derivation_key is not None:
            # Only the still-current failed generation may project a terminal
            # blocked state. A resolution receipt for an older generation must
            # not clear a newer pending mark.
            updated = conn.execute(
                """
                UPDATE embedding_derivation_state
                SET attempt_state = 'failed_terminal', updated_at_ms = ?
                WHERE session_id = ? AND generation = ? AND derivation_key = ?
                  AND attempt_state IN ('failed_retryable', 'failed_terminal')
                """,
                (now_ms, session_id, generation, derivation_key),
            )
            if updated.rowcount == 1:
                conn.execute(
                    "UPDATE embedding_status SET needs_reindex = 0 WHERE session_id = ? AND error_message IS NOT NULL",
                    (session_id,),
                )
        elif generation == 0:
            # A v2 receipt may outlive a rebuild/first keyed attempt. It may
            # update the audit ledger, but it can project status only while no
            # exact derivation generation exists for the session.
            has_derivation_state = _session_has_embedding_derivation_state(conn, session_id)
            if not has_derivation_state and action == "requeue":
                conn.execute(
                    "UPDATE embedding_status SET needs_reindex = 1, error_message = NULL WHERE session_id = ?",
                    (session_id,),
                )
            elif not has_derivation_state:
                conn.execute(
                    "UPDATE embedding_status SET needs_reindex = 0 WHERE session_id = ? AND error_message IS NOT NULL",
                    (session_id,),
                )
    return read_embedding_failure(conn, failure_id)


def resolve_open_embedding_failures_for_session(
    conn: sqlite3.Connection, *, session_id: str, resolved_at_ms: int | None = None
) -> int:
    """Preserve prior failures while marking a later successful embedding as resolution."""

    now_ms = int(time.time() * 1000) if resolved_at_ms is None else resolved_at_ms
    with conn:
        cursor = conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = 'resolved', updated_at_ms = ?, resolved_at_ms = ?, resolution_action = 'embedded'
            WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
            """,
            (now_ms, now_ms, session_id),
        )
    return max(0, cursor.rowcount)


def list_active_embedding_failures(conn: sqlite3.Connection, *, limit: int = 25) -> tuple[ArchiveEmbeddingFailure, ...]:
    """Return bounded current failure identities for status and agent surfaces."""

    identity_select = _embedding_failure_identity_select(conn)
    rows = conn.execute(
        f"""
        SELECT failure_id, session_id, origin, message_refs_json, provider, model, error_class, error_message,
               retryable, lifecycle_state, created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
               resolution_note, superseded_by, {identity_select}
        FROM embedding_failures
        WHERE lifecycle_state IN ('retryable', 'terminal')
        ORDER BY updated_at_ms DESC, failure_id ASC
        LIMIT ?
        """,
        (max(0, limit),),
    ).fetchall()
    return tuple(_failure_from_row(row) for row in rows)


def read_embedding_failure(conn: sqlite3.Connection, failure_id: str) -> ArchiveEmbeddingFailure:
    identity_select = _embedding_failure_identity_select(conn)
    row = conn.execute(
        f"""
        SELECT failure_id, session_id, origin, message_refs_json, provider, model, error_class, error_message,
               retryable, lifecycle_state, created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
               resolution_note, superseded_by, {identity_select}
        FROM embedding_failures WHERE failure_id = ?
        """,
        (failure_id,),
    ).fetchone()
    if row is None:
        raise KeyError(failure_id)
    return _failure_from_row(row)


def _session_has_embedding_derivation_state(conn: sqlite3.Connection, session_id: str) -> bool:
    """Return whether a keyed generation exists, tolerating pre-v3 fixtures."""

    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embedding_derivation_state'"
    ).fetchone()
    if table is None:
        return False
    return (
        conn.execute(
            "SELECT 1 FROM embedding_derivation_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        is not None
    )


def _embedding_failure_identity_select(conn: sqlite3.Connection) -> str:
    """Read v3 failure identity columns while tolerating rebuildable v2 fixtures."""

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(embedding_failures)").fetchall()}
    return ", ".join(
        (
            "generation" if "generation" in columns else "0 AS generation",
            "derivation_key" if "derivation_key" in columns else "NULL AS derivation_key",
            "source_hash" if "source_hash" in columns else "NULL AS source_hash",
            "recipe_hash" if "recipe_hash" in columns else "NULL AS recipe_hash",
        )
    )


def _failure_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveEmbeddingFailure:
    message_refs_raw = row[3]
    try:
        message_refs = tuple(str(item) for item in json.loads(str(message_refs_raw)))
    except (TypeError, ValueError, json.JSONDecodeError):
        message_refs = ()
    return ArchiveEmbeddingFailure(
        failure_id=str(row[0]),
        session_id=str(row[1]),
        origin=str(row[2]),
        message_refs=message_refs,
        provider=str(row[4]),
        model=str(row[5]),
        error_class=str(row[6]),
        error_message=str(row[7]),
        retryable=bool(row[8]),
        lifecycle_state=cast(EmbeddingFailureState, str(row[9])),
        created_at_ms=int(str(row[10])),
        updated_at_ms=int(str(row[11])),
        resolved_at_ms=None if row[12] is None else int(str(row[12])),
        resolution_action=None if row[13] is None else str(row[13]),
        resolution_note=None if row[14] is None else str(row[14]),
        superseded_by=None if row[15] is None else str(row[15]),
        generation=int(str(row[16])),
        derivation_key=None if row[17] is None else cast(bytes, row[17]),
        source_hash=None if row[18] is None else cast(bytes, row[18]),
        recipe_hash=None if row[19] is None else cast(bytes, row[19]),
    )


def read_embedding_meta(conn: sqlite3.Connection, target_id: str) -> ArchiveEmbeddingMeta:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex
        FROM message_embeddings_meta
        WHERE message_id = ?
        """,
        (target_id,),
    ).fetchone()
    if row is None:
        raise KeyError(target_id)
    return ArchiveEmbeddingMeta(
        message_id=row["message_id"],
        model=row["model"],
        dimension=row["dimension"],
        content_hash=row["content_hash"],
        embedded_at_ms=row["embedded_at_ms"],
        needs_reindex=bool(row["needs_reindex"]),
    )


def read_embedding_status(conn: sqlite3.Connection, session_id: str) -> ArchiveEmbeddingStatus:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
        FROM embedding_status
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(session_id)
    return ArchiveEmbeddingStatus(
        session_id=row["session_id"],
        origin=row["origin"],
        message_count_embedded=row["message_count_embedded"],
        last_embedded_at_ms=row["last_embedded_at_ms"],
        needs_reindex=bool(row["needs_reindex"]),
        error_message=row["error_message"],
    )


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


__all__ = [
    "ArchiveEmbeddingAttempt",
    "ArchiveEmbeddingFailure",
    "ArchiveEmbeddingMeta",
    "ArchiveEmbeddingStatus",
    "ArchiveEmbeddingWrite",
    "EmbeddingFailureResolution",
    "EmbeddingFailureState",
    "begin_embedding_attempt",
    "complete_embedding_attempt_success",
    "list_active_embedding_failures",
    "mark_session_embedding_error",
    "read_embedding_failure",
    "read_embedding_meta",
    "read_embedding_status",
    "record_embedding_failure",
    "resolve_embedding_failure",
    "resolve_open_embedding_failures_for_session",
    "supersede_embedding_attempt",
    "upsert_message_embedding",
    "upsert_message_embeddings",
]
