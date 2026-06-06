"""Per-session embedding similarity read surface (#1123).

Surfaces "find similar sessions" through the embedding pipeline
established by #828. The pipeline is dormant by default — most archives
have no embedded messages — so this module's job is as much about
explicit absent-state rendering as it is about ranked results.

Contract:

- The endpoint never embeds new content. It only reads vectors that have
  already been materialized by the daemon's embedding stage.
- When the operator has not enabled embeddings (``embedding_enabled`` is
  false or ``voyage_api_key`` is missing in ``polylogue.toml``), the
  endpoint returns ``status="disabled"`` with a machine-readable
  ``reason`` and an empty result list.
- When embeddings are enabled but the runtime is missing
  (``sqlite-vec`` not installed, or the ``message_embeddings`` table
  does not exist yet), the endpoint returns ``status="unavailable"``
  with the specific reason instead of pretending to return zero hits.
- When the source session has no embedded messages (the common
  case while a backlog is catching up), the endpoint returns
  ``status="not_embedded"`` with an empty result list. The caller
  should render the "this session is not yet embedded" state.
- Otherwise the endpoint returns ``status="ready"`` with a ranked list
  of session hits. Each hit carries a numeric ``score`` (cosine
  similarity in ``[0, 1]``, higher is more similar) and a coarse
  ``confidence`` chip key (``q-canonical`` / ``q-estimated`` /
  ``q-heuristic``) derived from the score band.

The endpoint reads only from existing tables — no API key is required to
serve a similarity lookup, because the source session already has
its vectors stored. This is the property that lets the reader expose
"find similar" without dragging the embedding provider back online.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from polylogue.config import load_polylogue_config
from polylogue.paths import active_index_db_path
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

# Hard server-side cap on requested result count. A pathological client
# asking for ``limit=10**6`` still receives at most this many rows.
SIMILAR_RESULTS_MAX: Final[int] = 50
SIMILAR_RESULTS_DEFAULT: Final[int] = 10

# Per-source-message neighbor fanout used to grow the candidate pool
# before deduplicating to sessions. Picked so the worst case
# (``limit=SIMILAR_RESULTS_MAX``) still completes in a single small
# ``MATCH`` per source message.
_PER_MESSAGE_K: Final[int] = 20


@dataclass(frozen=True)
class SimilarHit:
    """One ranked similar-session row."""

    session_id: str
    score: float
    confidence: str
    title: str | None
    source_name: str | None
    matched_message_count: int


def _confidence_for_score(score: float) -> str:
    """Map a cosine-similarity score to a coarse confidence chip key.

    The bands are intentionally wide. They are not asserting calibrated
    probabilities — they are giving the reader a stable, human-readable
    "how seriously should I take this row" cue that matches the rest of
    the MK3 ``q-*`` vocabulary already used by the cost panel.
    """

    if score >= 0.75:
        return "q-canonical"
    if score >= 0.55:
        return "q-estimated"
    return "q-heuristic"


def _disabled_reason(*, embedding_enabled: bool, voyage_api_key: str | None) -> str | None:
    """Return the explicit disabled-state reason, or ``None`` if enabled.

    The two failure modes are kept distinct so the reader can render
    actionable guidance — "set ``VOYAGE_API_KEY``" vs. "flip
    ``embedding_enabled`` in polylogue.toml" are different fixes.
    """

    if not embedding_enabled:
        return "embeddings_not_enabled"
    if not voyage_api_key:
        return "no_voyage_api_key"
    return None


def _empty_envelope(status: str, *, reason: str | None) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "session_id": None,
        "source_embedded_messages": 0,
        "limit": SIMILAR_RESULTS_DEFAULT,
        "results": [],
    }


def _fetch_session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _fetch_archive_session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _vec_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='message_embeddings'").fetchone()
    return row is not None


def _source_message_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
) -> list[tuple[str, bytes]]:
    """Return ``(message_id, embedding_blob)`` rows for the source session.

    The blob is returned exactly as stored in ``message_embeddings`` so
    it can be passed back into a ``vec0`` ``MATCH`` clause without any
    re-encoding round trip.
    """

    rows = conn.execute(
        """
        SELECT message_id, embedding
        FROM message_embeddings
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()
    return [(str(row[0]), bytes(row[1])) for row in rows]


def _source_archive_message_embeddings(
    conn: sqlite3.Connection,
    session_id: str,
) -> list[tuple[str, bytes]]:
    rows = conn.execute(
        """
        SELECT message_id, embedding
        FROM message_embeddings
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchall()
    return [(str(row[0]), bytes(row[1])) for row in rows]


def _knn_for_embedding(
    conn: sqlite3.Connection,
    embedding_blob: bytes,
    *,
    k: int,
) -> list[tuple[str, str, float]]:
    """Run a single ``MATCH`` and return ``(message_id, session_id, distance)``.

    The vec0 column ``distance`` here is L2 distance over the indexed
    embeddings. Cosine similarity is recovered downstream once a row's
    embedding has been pulled into the score-aggregation step.
    """

    rows = conn.execute(
        """
        SELECT message_id, session_id, distance
        FROM message_embeddings
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (embedding_blob, k),
    ).fetchall()
    return [(str(row[0]), str(row[1]), float(row[2])) for row in rows]


def _archive_knn_for_embedding(
    conn: sqlite3.Connection,
    embedding_blob: bytes,
    *,
    k: int,
) -> list[tuple[str, str, float]]:
    rows = conn.execute(
        """
        SELECT message_id, session_id, distance
        FROM message_embeddings
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance
        """,
        (embedding_blob, k),
    ).fetchall()
    return [(str(row[0]), str(row[1]), float(row[2])) for row in rows]


def _l2_to_cosine_similarity(distance: float) -> float:
    """Convert vec0 L2 distance over unit-norm vectors to cosine similarity.

    Voyage embeddings are L2-normalized, so for unit vectors
    ``||a - b||^2 = 2 - 2 cos(theta)`` and therefore
    ``cos(theta) = 1 - distance^2 / 2``. The result is clamped to
    ``[0, 1]`` so noise from non-unit-norm rows can never produce a
    confidence higher than the upper band.
    """

    sim = 1.0 - (distance * distance) / 2.0
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


def _aggregate_hits(
    per_message_results: list[list[tuple[str, str, float]]],
    *,
    self_session_id: str,
) -> dict[str, dict[str, float | int]]:
    """Collapse per-message KNN lists into per-session aggregates.

    Each candidate session keeps its best (closest) distance seen
    across all source messages. The matched-message count records how
    many source messages picked it as a neighbor — a coarse but useful
    "this isn't a one-off accidental hit" signal exposed to the reader.
    """

    aggregates: dict[str, dict[str, float | int]] = {}
    for hits in per_message_results:
        seen_in_this_source: set[str] = set()
        for _msg_id, conv_id, distance in hits:
            if conv_id == self_session_id:
                continue
            entry = aggregates.setdefault(
                conv_id,
                {"best_distance": float("inf"), "matched_messages": 0},
            )
            if distance < entry["best_distance"]:
                entry["best_distance"] = distance
            if conv_id not in seen_in_this_source:
                seen_in_this_source.add(conv_id)
                entry["matched_messages"] = int(entry["matched_messages"]) + 1
    return aggregates


def _hydrate_session_metadata(
    conn: sqlite3.Connection,
    session_ids: list[str],
) -> dict[str, tuple[str | None, str | None]]:
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    rows = conn.execute(
        f"""
        SELECT session_id, title, source_name
        FROM sessions
        WHERE session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    return {
        str(row[0]): (
            (str(row[1]) if row[1] is not None else None),
            (str(row[2]) if row[2] is not None else None),
        )
        for row in rows
    }


def _hydrate_archive_session_metadata(
    conn: sqlite3.Connection,
    session_ids: list[str],
) -> dict[str, tuple[str | None, str | None]]:
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    rows = conn.execute(
        f"""
        SELECT session_id, title, origin
        FROM sessions
        WHERE session_id IN ({placeholders})
        """,
        tuple(session_ids),
    ).fetchall()
    return {
        str(row[0]): (
            (str(row[1]) if row[1] is not None else None),
            (str(row[2]) if row[2] is not None else None),
        )
        for row in rows
    }


def _clamp_limit(requested: int | None) -> int:
    if requested is None:
        return SIMILAR_RESULTS_DEFAULT
    if requested <= 0:
        return SIMILAR_RESULTS_DEFAULT
    if requested > SIMILAR_RESULTS_MAX:
        return SIMILAR_RESULTS_MAX
    return requested


def _archive_index_path_for() -> str | None:
    path = active_index_db_path()
    if path.name != "index.db":
        path = path.with_name("index.db")
    if path.exists():
        return str(path)
    return None


def _build_archive_similar_payload(
    index_db: str,
    session_id: str,
    *,
    bounded_limit: int,
    disabled_reason: str | None,
) -> dict[str, object] | None:
    conn: sqlite3.Connection | None = None
    index_conn = sqlite3.connect(index_db)
    try:
        index_conn.row_factory = sqlite3.Row
        if not _fetch_archive_session_exists(index_conn, session_id):
            return None

        if disabled_reason is not None:
            envelope = _empty_envelope("disabled", reason=disabled_reason)
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        embeddings_db = Path(index_db).with_name("embeddings.db")
        if not embeddings_db.exists():
            envelope = _empty_envelope("unavailable", reason="vec0_table_missing")
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope
        conn = sqlite3.connect(str(embeddings_db))
        conn.row_factory = sqlite3.Row
        if not _vec_table_exists(conn):
            envelope = _empty_envelope("unavailable", reason="vec0_table_missing")
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        loaded, _ext_error = try_load_sqlite_vec(conn)
        if not loaded:
            envelope = _empty_envelope("unavailable", reason="sqlite_vec_not_loaded")
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        source_rows = _source_archive_message_embeddings(conn, session_id)
        if not source_rows:
            envelope = _empty_envelope("not_embedded", reason=None)
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        per_message_results: list[list[tuple[str, str, float]]] = []
        for _msg_id, embedding in source_rows[:_PER_MESSAGE_K]:
            per_message_results.append(_archive_knn_for_embedding(conn, embedding, k=_PER_MESSAGE_K))

        aggregates = _aggregate_hits(
            per_message_results,
            self_session_id=session_id,
        )
        ranked = sorted(
            aggregates.items(),
            key=lambda item: (item[1]["best_distance"], item[0]),
        )[:bounded_limit]
        metadata = _hydrate_archive_session_metadata(
            index_conn,
            [session_id for session_id, _ in ranked],
        )
        hits: list[dict[str, object]] = []
        for session_id, agg in ranked:
            distance = float(agg["best_distance"])
            score = _l2_to_cosine_similarity(distance)
            title, source_name = metadata.get(session_id, (None, None))
            hits.append(
                {
                    "session_id": session_id,
                    "score": round(score, 4),
                    "distance": round(distance, 4),
                    "confidence": _confidence_for_score(score),
                    "title": title,
                    "source_name": source_name,
                    "matched_message_count": int(agg["matched_messages"]),
                }
            )

        return {
            "status": "ready",
            "reason": None,
            "session_id": session_id,
            "source_embedded_messages": len(source_rows),
            "limit": bounded_limit,
            "results": hits,
        }
    finally:
        if conn is not None:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        index_conn.close()


def build_similar_payload(
    session_id: str,
    *,
    limit: int | None = None,
) -> dict[str, object] | None:
    """Assemble the JSON payload for ``GET /api/sessions/{id}/similar``.

    Returns ``None`` when the session does not exist so the caller
    can emit a 404. Returns a structured envelope in every other case;
    "no results" is never silent.

    The envelope ``status`` field is one of:

    - ``"disabled"`` — embeddings not enabled or no Voyage API key.
    - ``"unavailable"`` — embeddings are enabled but the ``vec0`` table
      or the ``sqlite-vec`` extension is missing.
    - ``"not_embedded"`` — the source session has no message
      vectors stored yet (waiting on the embedding pipeline).
    - ``"ready"`` — ranked similar sessions are attached under
      ``results``. ``results`` may still be empty if no other
      session shares an embedded neighbor.
    """

    bounded_limit = _clamp_limit(limit)
    cfg = load_polylogue_config()
    disabled_reason = _disabled_reason(
        embedding_enabled=bool(cfg.embedding_enabled),
        voyage_api_key=cfg.voyage_api_key,
    )

    dbf = active_index_db_path()
    index_db = _archive_index_path_for()
    if index_db is not None:
        return _build_archive_similar_payload(
            index_db,
            session_id,
            bounded_limit=bounded_limit,
            disabled_reason=disabled_reason,
        )
    if not dbf.exists():
        # Treat a missing database the same as a missing session —
        # the route layer turns this into 404. The reader will never see
        # this branch in practice once the archive has been bootstrapped.
        return None

    conn = sqlite3.connect(str(dbf))
    try:
        conn.row_factory = sqlite3.Row
        if not _fetch_session_exists(conn, session_id):
            return None

        if disabled_reason is not None:
            envelope = _empty_envelope("disabled", reason=disabled_reason)
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        if not _vec_table_exists(conn):
            envelope = _empty_envelope("unavailable", reason="vec0_table_missing")
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        loaded, _ext_error = try_load_sqlite_vec(conn)
        if not loaded:
            envelope = _empty_envelope("unavailable", reason="sqlite_vec_not_loaded")
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        source_rows = _source_message_embeddings(conn, session_id)
        if not source_rows:
            envelope = _empty_envelope("not_embedded", reason=None)
            envelope["session_id"] = session_id
            envelope["limit"] = bounded_limit
            return envelope

        # Cap source fanout to avoid issuing a thousand MATCH queries
        # for a giant session. Twenty representative messages is
        # enough to populate the candidate pool with good recall for
        # the reader's top-N display.
        per_message_results: list[list[tuple[str, str, float]]] = []
        for _msg_id, embedding in source_rows[:_PER_MESSAGE_K]:
            per_message_results.append(_knn_for_embedding(conn, embedding, k=_PER_MESSAGE_K))

        aggregates = _aggregate_hits(
            per_message_results,
            self_session_id=session_id,
        )

        ranked = sorted(
            aggregates.items(),
            key=lambda item: (item[1]["best_distance"], item[0]),
        )[:bounded_limit]

        metadata = _hydrate_session_metadata(
            conn,
            [conv_id for conv_id, _ in ranked],
        )

        hits: list[dict[str, object]] = []
        for conv_id, agg in ranked:
            distance = float(agg["best_distance"])
            score = _l2_to_cosine_similarity(distance)
            title, source_name = metadata.get(conv_id, (None, None))
            hits.append(
                {
                    "session_id": conv_id,
                    "score": round(score, 4),
                    "distance": round(distance, 4),
                    "confidence": _confidence_for_score(score),
                    "title": title,
                    "source_name": source_name,
                    "matched_message_count": int(agg["matched_messages"]),
                }
            )

        return {
            "status": "ready",
            "reason": None,
            "session_id": session_id,
            "source_embedded_messages": len(source_rows),
            "limit": bounded_limit,
            "results": hits,
        }
    finally:
        conn.close()


__all__ = [
    "SIMILAR_RESULTS_DEFAULT",
    "SIMILAR_RESULTS_MAX",
    "SimilarHit",
    "build_similar_payload",
]
