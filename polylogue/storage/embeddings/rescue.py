"""Rescue vectors from a retired embeddings tier, landing directly into the
content-addressed (embedding_input_hash) layout.

polylogue-04kl: a prior incident retired an embeddings tier
(``embeddings.db.v2-retired-YYYYMMDD``) while the index was rebuilt from
source. That retired tier still holds hundreds of thousands of Voyage
vectors, keyed by the OLD v3 identity: ``message_id`` + a 32-byte
identity-contaminated ``content_hash`` (``messages.content_hash`` includes
session_id/position/variant_index -- see polylogue-q88p). The v4 embeddings
tier this rescue writes into is keyed by ``embedding_input_hash`` instead
(identity-free: ``H(model, embedder input text)``), so a straight copy is not
possible; rescue must *recompute* the new key for each candidate message and
land the vector under that key.

A retired vector is safe to rescue for a message whenever:

* its ``message_id`` still exists among the fresh index's *currently
  eligible* (authored-prose) messages,
* its stored ``content_hash`` matches that message's current content hash
  exactly (identity-contaminated, but an exact match remains a valid --
  merely conservative -- sufficient proof that the message's text is
  unchanged from what the retired vector was computed for), and
* its stored ``model`` matches the configured embedding model.

Once matched, this rescue recomputes ``embedding_input_hash`` from the
message's *current* embedder input text (the same prose expression the live
embed path uses) and writes the retired vector under that new key, plus a
``message_embedding_refs`` row for the message. The content_hash match is
what licenses treating "current text" and "retired text" as identical; the
new key is never trusted from the retired file (which cannot know it).

Session-level granularity is not a stylistic choice -- it is required by how
catch-up materialization actually works. ``embed_archive_session_sync``
(:mod:`polylogue.storage.embeddings.materialization`) always re-embeds
*every* eligible message of a session it selects, via one atomic
generation-guarded write (:func:`polylogue.storage.sqlite.archive_tiers.
embedding_write.complete_embedding_attempt_success`); it never consults
per-message vectors already present to skip a subset. So a session only
avoids a live API re-embed when *all* of its eligible messages are
rescuable -- a partial per-message match saves nothing, because the next
catch-up pass deletes and rewrites the whole session's vectors regardless.
Rescue therefore classifies and mutates at session granularity: a session
is ``fully_rescuable`` only when every one of its eligible messages has an
exact retired match; anything less is reported as a ``partial`` session and
left untouched (the real embed pass will cover it later at full API cost).

Rescued sessions are published through the same guarded primitives the live
embed path uses (:func:`begin_embedding_attempt` +
:func:`complete_embedding_attempt_success`), so a rescued session's
``embedding_derivation_state``/``embedding_status`` rows look identical to
one embedded for real -- the daemon's freshness predicate
(``_archive_embedding_freshness_predicate``) will no longer select it, and a
second rescue run naturally treats it as already-fresh (idempotent, no
re-write). Content-addressing also means a rescued vector is immediately
shared with every other message (any origin/session) whose current text
hashes to the same ``embedding_input_hash`` -- dedup applies to rescue too.
"""

from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.storage.embeddings.identity import (
    EmbeddingRecipe,
    EmbeddingSourceDigest,
    message_embedding_derivation_key,
)
from polylogue.storage.embeddings.materialization import (
    archive_embeddable_messages_relation,
    select_pending_archive_session_window,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

EmbeddingRescueMutationAuthority = Literal["offline-exclusive"]

DEFAULT_SAMPLE_SIZE = 30
DEFAULT_SAMPLE_VERIFY_COUNT = 20

_MISSING = "missing"
_MODEL_MISMATCH = "model_mismatch"
_HASH_MISMATCH = "hash_mismatch"
_MATCHED = "matched"

_CLASSIFIED_CTE_TEMPLATE = """
WITH eligible AS (
    SELECT e.message_id, e.session_id, e.content_hash
    FROM {relation}
),
classified AS (
    SELECT
        eligible.session_id AS session_id,
        eligible.message_id AS message_id,
        CASE
            WHEN rm.message_id IS NULL THEN '{missing}'
            WHEN rv.id IS NULL THEN '{missing}'
            WHEN rm.model != :model THEN '{model_mismatch}'
            WHEN eligible.content_hash IS NULL THEN '{hash_mismatch}'
            WHEN rm.content_hash != eligible.content_hash THEN '{hash_mismatch}'
            ELSE '{matched}'
        END AS status
    FROM eligible
    LEFT JOIN retired.message_embeddings_meta AS rm ON rm.message_id = eligible.message_id
    LEFT JOIN retired.message_embeddings_rowids AS rv ON rv.id = eligible.message_id
)
"""


def _classified_cte(relation: str) -> str:
    return _CLASSIFIED_CTE_TEMPLATE.format(
        relation=relation,
        missing=_MISSING,
        model_mismatch=_MODEL_MISMATCH,
        hash_mismatch=_HASH_MISMATCH,
        matched=_MATCHED,
    )


@dataclass(frozen=True, slots=True)
class EmbeddingRescueSample:
    """One representative message classification surfaced for operator inspection."""

    status: str  # "missing" | "hash_mismatch" | "model_mismatch"
    message_id: str
    session_id: str

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "message_id": self.message_id, "session_id": self.session_id}


@dataclass(frozen=True, slots=True)
class EmbeddingRescueCounts:
    """Shared session/message classification counts for plan and execute reports."""

    eligible_sessions: int
    fully_rescuable_sessions: int
    rescuable_messages: int
    partial_sessions: int
    partial_matched_messages: int
    skipped_missing: int
    skipped_hash_mismatch: int
    skipped_model_mismatch: int


@dataclass(frozen=True, slots=True)
class EmbeddingRescuePlanReport:
    """Read-only rescue-candidate census: ``ops maintenance embeddings-rescue --plan``."""

    index_db: str
    source_db: str
    model: str
    counts: EmbeddingRescueCounts
    samples: tuple[EmbeddingRescueSample, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "plan",
            "index_db": self.index_db,
            "source_db": self.source_db,
            "model": self.model,
            "eligible_sessions": self.counts.eligible_sessions,
            "fully_rescuable_sessions": self.counts.fully_rescuable_sessions,
            "rescuable_messages": self.counts.rescuable_messages,
            "partial_sessions": self.counts.partial_sessions,
            "partial_matched_messages": self.counts.partial_matched_messages,
            "skipped_missing": self.counts.skipped_missing,
            "skipped_hash_mismatch": self.counts.skipped_hash_mismatch,
            "skipped_model_mismatch": self.counts.skipped_model_mismatch,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class EmbeddingRescueExecuteReport:
    """Apply-mode rescue outcome: ``ops maintenance embeddings-rescue --yes``."""

    index_db: str
    source_db: str
    embeddings_db: str
    model: str
    counts: EmbeddingRescueCounts
    rescued_sessions: int
    rescued_messages: int
    skipped_already_fresh_sessions: int
    skipped_race_sessions: int
    verified_sample_total: int
    verified_sample_ok: int
    more_pending: bool
    samples: tuple[EmbeddingRescueSample, ...]

    @property
    def ok(self) -> bool:
        """Whether every sampled rescued vector verified byte-identical to its source."""
        return self.verified_sample_total == 0 or self.verified_sample_ok == self.verified_sample_total

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": "execute",
            "ok": self.ok,
            "index_db": self.index_db,
            "source_db": self.source_db,
            "embeddings_db": self.embeddings_db,
            "model": self.model,
            "eligible_sessions": self.counts.eligible_sessions,
            "fully_rescuable_sessions": self.counts.fully_rescuable_sessions,
            "rescuable_messages": self.counts.rescuable_messages,
            "partial_sessions": self.counts.partial_sessions,
            "partial_matched_messages": self.counts.partial_matched_messages,
            "skipped_missing": self.counts.skipped_missing,
            "skipped_hash_mismatch": self.counts.skipped_hash_mismatch,
            "skipped_model_mismatch": self.counts.skipped_model_mismatch,
            "rescued_sessions": self.rescued_sessions,
            "rescued_messages": self.rescued_messages,
            "skipped_already_fresh_sessions": self.skipped_already_fresh_sessions,
            "skipped_race_sessions": self.skipped_race_sessions,
            "verified_sample_total": self.verified_sample_total,
            "verified_sample_ok": self.verified_sample_ok,
            "more_pending": self.more_pending,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def default_embedding_recipe() -> EmbeddingRecipe:
    from polylogue.config import load_polylogue_config
    from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION

    cfg = load_polylogue_config()
    return EmbeddingRecipe.current(model=str(cfg.embedding_model), dimensions=EMBEDDING_DIMENSION)


def _open_classification_connection(index_db_path: Path, source_db_path: Path) -> sqlite3.Connection:
    if not source_db_path.exists():
        raise RuntimeError(f"retired embeddings source not found: {source_db_path}")
    conn = open_readonly_connection(index_db_path, timeout=30.0)
    try:
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("embedding rescue requires sqlite-vec") from error
        conn.execute("ATTACH DATABASE ? AS retired", (f"file:{source_db_path}?mode=ro",))
    except BaseException:
        conn.close()
        raise
    return conn


def _session_rollup_counts(conn: sqlite3.Connection, relation: str, model: str) -> EmbeddingRescueCounts:
    session_rows = conn.execute(
        f"""
        {_classified_cte(relation)},
        session_rollup AS (
            SELECT session_id, COUNT(*) AS eligible_count, SUM(status = '{_MATCHED}') AS matched_count
            FROM classified
            GROUP BY session_id
        )
        SELECT
            COUNT(*) AS eligible_sessions,
            COALESCE(SUM(CASE WHEN matched_count = eligible_count THEN 1 ELSE 0 END), 0) AS fully_rescuable_sessions,
            COALESCE(SUM(CASE WHEN matched_count = eligible_count THEN matched_count ELSE 0 END), 0) AS rescuable_messages,
            COALESCE(
                SUM(CASE WHEN matched_count > 0 AND matched_count < eligible_count THEN 1 ELSE 0 END), 0
            ) AS partial_sessions,
            COALESCE(
                SUM(CASE WHEN matched_count > 0 AND matched_count < eligible_count THEN matched_count ELSE 0 END), 0
            ) AS partial_matched_messages
        FROM session_rollup
        """,
        {"model": model},
    ).fetchone()

    status_rows = conn.execute(
        f"""
        {_classified_cte(relation)}
        SELECT status, COUNT(*) FROM classified WHERE status != '{_MATCHED}' GROUP BY status
        """,
        {"model": model},
    ).fetchall()
    status_counts = {str(row[0]): int(row[1]) for row in status_rows}

    return EmbeddingRescueCounts(
        eligible_sessions=int(session_rows[0] or 0),
        fully_rescuable_sessions=int(session_rows[1] or 0),
        rescuable_messages=int(session_rows[2] or 0),
        partial_sessions=int(session_rows[3] or 0),
        partial_matched_messages=int(session_rows[4] or 0),
        skipped_missing=status_counts.get(_MISSING, 0),
        skipped_hash_mismatch=status_counts.get(_HASH_MISMATCH, 0),
        skipped_model_mismatch=status_counts.get(_MODEL_MISMATCH, 0),
    )


def _classification_samples(
    conn: sqlite3.Connection, relation: str, model: str, *, sample_size: int
) -> tuple[EmbeddingRescueSample, ...]:
    if sample_size <= 0:
        return ()
    rows = conn.execute(
        f"""
        {_classified_cte(relation)}
        SELECT status, message_id, session_id
        FROM classified
        WHERE status != '{_MATCHED}'
        ORDER BY message_id
        LIMIT :limit
        """,
        {"model": model, "limit": sample_size},
    ).fetchall()
    return tuple(
        EmbeddingRescueSample(status=str(row[0]), message_id=str(row[1]), session_id=str(row[2])) for row in rows
    )


def _fully_rescuable_session_ids(conn: sqlite3.Connection, relation: str, model: str) -> tuple[str, ...]:
    rows = conn.execute(
        f"""
        {_classified_cte(relation)},
        session_rollup AS (
            SELECT session_id, COUNT(*) AS eligible_count, SUM(status = '{_MATCHED}') AS matched_count
            FROM classified
            GROUP BY session_id
        )
        SELECT session_id FROM session_rollup WHERE matched_count = eligible_count AND eligible_count > 0
        """,
        {"model": model},
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def plan_embedding_rescue(
    index_db_path: str | Path,
    source_embeddings_db_path: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    recipe: EmbeddingRecipe | None = None,
) -> EmbeddingRescuePlanReport:
    """Read-only census of vectors rescuable from a retired embeddings tier.

    Never mutates either database; safe to run against a live archive
    (including mid-rebuild) and against the read-only retired evidence file.
    """
    index_path = Path(index_db_path)
    source_path = Path(source_embeddings_db_path)
    resolved_recipe = recipe or default_embedding_recipe()

    conn = _open_classification_connection(index_path, source_path)
    try:
        relation = archive_embeddable_messages_relation(conn, alias="e")
        counts = _session_rollup_counts(conn, relation, resolved_recipe.model)
        samples = _classification_samples(conn, relation, resolved_recipe.model, sample_size=sample_size)
    finally:
        conn.close()

    return EmbeddingRescuePlanReport(
        index_db=str(index_path),
        source_db=str(source_path),
        model=resolved_recipe.model,
        counts=counts,
        samples=samples,
    )


def _session_eligible_messages(
    conn: sqlite3.Connection, relation: str, session_id: str
) -> list[tuple[str, bytes, bytes]]:
    """Return ``(message_id, content_hash, embedding_input_hash)`` triples.

    ``content_hash`` licenses the retired-vector match (old, identity-
    contaminated identity); ``embedding_input_hash`` -- computed by the
    relation from the message's *current* embedder input text -- is the key
    the rescued vector is written under in the new content-addressed layout.
    """
    rows = conn.execute(
        f"SELECT e.message_id, e.content_hash, e.embedding_input_hash FROM {relation} WHERE e.session_id = ?",
        (session_id,),
    ).fetchall()
    return [(str(row[0]), bytes(row[1]), bytes(row[2])) for row in rows]


def _source_hash_for(messages: list[tuple[str, bytes, bytes]]) -> bytes:
    digest = EmbeddingSourceDigest()
    for _message_id, _content_hash, input_hash in sorted(messages, key=lambda item: item[2]):
        digest.update(input_hash)
    return digest.digest()


def _read_retired_vector(conn: sqlite3.Connection, message_id: str) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM retired.message_embeddings WHERE message_id = ?",
        (message_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    raw = bytes(row[0])
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw))


def execute_embedding_rescue(
    index_db_path: str | Path,
    source_embeddings_db_path: str | Path,
    embeddings_db_path: str | Path | None = None,
    *,
    limit: int | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_verify_count: int = DEFAULT_SAMPLE_VERIFY_COUNT,
    recipe: EmbeddingRecipe | None = None,
    mutation_authority: EmbeddingRescueMutationAuthority | None = None,
    now_ms: int | None = None,
) -> EmbeddingRescueExecuteReport:
    """Copy vectors for fully-rescuable sessions from the retired tier.

    A session is only mutated when *every* one of its currently-eligible
    messages has an exact retired match (see module docstring for why
    partial sessions are never worth writing). Publication goes through
    :func:`begin_embedding_attempt` / :func:`complete_embedding_attempt_success`
    -- the same generation-guarded primitives the live embed path uses -- so
    a rescued session is indistinguishable from one embedded for real, and a
    second invocation is a no-op over already-rescued sessions (idempotent,
    resumable via ``limit``).
    """
    if mutation_authority is None:
        raise RuntimeError("embedding rescue apply requires offline-exclusive authority")

    index_path = Path(index_db_path)
    source_path = Path(source_embeddings_db_path)
    embeddings_path = (
        Path(embeddings_db_path) if embeddings_db_path is not None else index_path.with_name("embeddings.db")
    )
    resolved_recipe = recipe or default_embedding_recipe()
    resolved_now_ms = now_ms if now_ms is not None else int(time.time() * 1000)

    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        ArchiveEmbeddingWrite,
        begin_embedding_attempt,
        complete_embedding_attempt_success,
        supersede_embedding_attempt,
    )
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_path, ArchiveTier.EMBEDDINGS)

    index_conn = _open_classification_connection(index_path, source_path)
    embeddings_conn = sqlite3.connect(embeddings_path, timeout=30.0)
    rescued_sessions = 0
    rescued_messages = 0
    race_skipped = 0
    already_fresh_count = 0
    more_pending = False
    verified_total = 0
    verified_ok = 0
    try:
        loaded, error = try_load_sqlite_vec(embeddings_conn)
        if not loaded:
            raise RuntimeError("embedding rescue requires sqlite-vec") from error

        relation = archive_embeddable_messages_relation(index_conn, alias="e", model=resolved_recipe.model)
        counts = _session_rollup_counts(index_conn, relation, resolved_recipe.model)
        samples = _classification_samples(index_conn, relation, resolved_recipe.model, sample_size=sample_size)
        fully_rescuable_ids = _fully_rescuable_session_ids(index_conn, relation, resolved_recipe.model)

        index_conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_path),))
        to_rescue = select_pending_archive_session_window(
            index_conn,
            status_table="embeddings.embedding_status",
            session_ids=fully_rescuable_ids,
            max_sessions=limit,
            recipe=resolved_recipe,
        )
        if fully_rescuable_ids:
            total_pending = select_pending_archive_session_window(
                index_conn,
                status_table="embeddings.embedding_status",
                session_ids=fully_rescuable_ids,
                recipe=resolved_recipe,
            )
            already_fresh_count = len(fully_rescuable_ids) - len(total_pending)
            more_pending = len(total_pending) > len(to_rescue)

        origin_by_session: dict[str, str] = {}
        if to_rescue:
            placeholders = ", ".join("?" for _ in to_rescue)
            origin_by_session = {
                str(row[0]): str(row[1])
                for row in index_conn.execute(
                    f"SELECT session_id, origin FROM sessions WHERE session_id IN ({placeholders})",
                    tuple(item.session_id for item in to_rescue),
                ).fetchall()
            }

        rescued_message_ids: list[str] = []

        for pending in to_rescue:
            session_id = pending.session_id
            origin = origin_by_session.get(session_id)
            if origin is None:
                continue
            messages = _session_eligible_messages(index_conn, relation, session_id)
            if not messages:
                continue
            source_hash = _source_hash_for(messages)

            attempt = begin_embedding_attempt(
                embeddings_conn,
                session_id=session_id,
                origin=origin,
                source_hash=source_hash,
                recipe=resolved_recipe,
            )

            writes: list[ArchiveEmbeddingWrite] = []
            incomplete = False
            for message_id, _content_hash, input_hash in messages:
                vector = _read_retired_vector(index_conn, message_id)
                if vector is None:
                    incomplete = True
                    break
                writes.append(
                    ArchiveEmbeddingWrite(
                        message_id=message_id,
                        session_id=session_id,
                        origin=origin,
                        embedding=vector,
                        model=resolved_recipe.model,
                        embedded_at_ms=resolved_now_ms,
                        embedding_input_hash=input_hash,
                        recipe_hash=attempt.recipe_hash,
                        derivation_key=message_embedding_derivation_key(
                            message_id=message_id,
                            embedding_input_hash=input_hash,
                            recipe=resolved_recipe,
                        ).digest(),
                        generation=attempt.generation,
                    )
                )

            if incomplete:
                # Classified as fully rescuable but the retired vector vanished
                # between classification and read (corrupt/edited retired
                # evidence). Never publish a partial write; requeue for the
                # real embed path instead of leaving a half-written attempt.
                supersede_embedding_attempt(
                    embeddings_conn,
                    attempt=attempt,
                    source_hash=source_hash,
                    recipe=resolved_recipe,
                )
                race_skipped += 1
                continue

            current_messages = _session_eligible_messages(index_conn, relation, session_id)
            current_source_hash = _source_hash_for(current_messages)
            if current_source_hash != source_hash or len(current_messages) != len(messages):
                supersede_embedding_attempt(
                    embeddings_conn,
                    attempt=attempt,
                    source_hash=current_source_hash,
                    recipe=resolved_recipe,
                )
                race_skipped += 1
                continue

            committed = complete_embedding_attempt_success(
                embeddings_conn,
                attempt=attempt,
                writes=writes,
                completed_at_ms=resolved_now_ms,
            )
            if not committed:
                race_skipped += 1
                continue

            rescued_sessions += 1
            rescued_messages += len(writes)
            rescued_message_ids.extend(message_id for message_id, _content_hash, _input_hash in messages)

        if rescued_message_ids and sample_verify_count > 0:
            step = max(1, len(rescued_message_ids) // sample_verify_count)
            sample_ids = rescued_message_ids[::step][:sample_verify_count]
            for message_id in sample_ids:
                verified_total += 1
                ref_row = embeddings_conn.execute(
                    "SELECT embedding_input_hash FROM message_embedding_refs WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if ref_row is None:
                    continue
                target_row = embeddings_conn.execute(
                    "SELECT embedding FROM message_embeddings WHERE embedding_input_hash = ?",
                    (bytes(ref_row[0]).hex(),),
                ).fetchone()
                source_row = index_conn.execute(
                    "SELECT embedding FROM retired.message_embeddings WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if target_row is not None and source_row is not None and bytes(target_row[0]) == bytes(source_row[0]):
                    verified_ok += 1
    finally:
        index_conn.close()
        embeddings_conn.close()

    return EmbeddingRescueExecuteReport(
        index_db=str(index_path),
        source_db=str(source_path),
        embeddings_db=str(embeddings_path),
        model=resolved_recipe.model,
        counts=counts,
        rescued_sessions=rescued_sessions,
        rescued_messages=rescued_messages,
        skipped_already_fresh_sessions=already_fresh_count,
        skipped_race_sessions=race_skipped,
        verified_sample_total=verified_total,
        verified_sample_ok=verified_ok,
        more_pending=more_pending,
        samples=samples,
    )


__all__ = [
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SAMPLE_VERIFY_COUNT",
    "EmbeddingRescueCounts",
    "EmbeddingRescueExecuteReport",
    "EmbeddingRescueMutationAuthority",
    "EmbeddingRescuePlanReport",
    "EmbeddingRescueSample",
    "default_embedding_recipe",
    "execute_embedding_rescue",
    "plan_embedding_rescue",
]
