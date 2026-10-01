"""Per-session embedding similarity read surface (#1123).

Surfaces "find similar sessions" through the embedding pipeline
established by #828. The pipeline is dormant by default — most archives
have no embedded messages — so this module's job is as much about
explicit absent-state rendering as it is about ranked results.

Contract:

- The endpoint never embeds new content. It only reads vectors that have
  already been materialized by the daemon's embedding stage.
- When the operator has not enabled embeddings (``embedding_enabled`` is
  false in ``polylogue.toml``), the
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
- When the vector store returns neighbors whose message ids exist in no
  indexed row, the ranking join is broken rather than empty; the endpoint
  returns ``status="inconsistent"`` with
  ``reason="embedded_messages_missing_from_index"`` instead of an
  indistinguishable empty ``ready``.
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

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Final, cast

from polylogue.config import Config, load_polylogue_config
from polylogue.core.errors import SessionNotFoundError, VectorReadUnavailableError
from polylogue.core.sqlite_introspection import table_exists
from polylogue.daemon.status import open_readonly_connection
from polylogue.operations.vector_reads import read_retained_vectors
from polylogue.paths import archive_root
from polylogue.storage.archive_identity import resolve_active_index_path

# Hard server-side cap on requested result count. A pathological client
# asking for ``limit=10**6`` still receives at most this many rows.
SIMILAR_RESULTS_MAX: Final[int] = 50
SIMILAR_RESULTS_DEFAULT: Final[int] = 10


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


def _disabled_reason(*, embedding_enabled: bool) -> str | None:
    """Retained vector reads depend on policy, never acquisition credentials."""
    return None if embedding_enabled else "embeddings_not_enabled"


def _empty_envelope(status: str, *, reason: str | None) -> dict[str, object]:
    return {
        "status": status,
        "reason": reason,
        "session_id": None,
        "source_embedded_messages": 0,
        "limit": SIMILAR_RESULTS_DEFAULT,
        "results": [],
    }


def _fetch_archive_session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _clamp_limit(requested: int | None) -> int:
    if requested is None:
        return SIMILAR_RESULTS_DEFAULT
    if requested <= 0:
        return SIMILAR_RESULTS_DEFAULT
    if requested > SIMILAR_RESULTS_MAX:
        return SIMILAR_RESULTS_MAX
    return requested


def _build_archive_similar_payload(
    index_db: str,
    session_id: str,
    *,
    bounded_limit: int,
    disabled_reason: str | None,
    archive_root_path: Path,
    embedding_model: str,
    embedding_dimension: int,
) -> dict[str, object] | None:
    if disabled_reason is not None:
        with closing(open_readonly_connection(index_db, timeout_class="interactive-read")) as index_conn:
            if not _fetch_archive_session_exists(index_conn, session_id):
                return None
        envelope = _empty_envelope("disabled", reason=disabled_reason)
        envelope["session_id"] = session_id
        envelope["limit"] = bounded_limit
        return envelope

    def read() -> dict[str, object]:
        embeddings_db = archive_root_path / "embeddings.db"
        if not embeddings_db.exists():
            raise VectorReadUnavailableError("vector table is absent", reason="vec0_table_missing")
        with open_readonly_connection(embeddings_db, timeout_class="interactive-read") as conn:
            if not table_exists(conn, "message_embeddings"):
                raise VectorReadUnavailableError("vector table is absent", reason="vec0_table_missing")

        from polylogue import Polylogue
        from polylogue.api.sync.bridge import run_coroutine_sync

        async def query() -> dict[str, object]:
            config = Config(
                archive_root=archive_root_path,
                render_root=archive_root_path / "render",
                sources=[],
                db_path=Path(index_db).resolve(strict=True),
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
            async with Polylogue(config=config) as polylogue:
                return await polylogue.search_similar_sessions(
                    session_id,
                    limit=bounded_limit,
                )

        return run_coroutine_sync(query())

    try:
        query_result = read_retained_vectors(read)
    except SessionNotFoundError:
        return None
    except VectorReadUnavailableError as exc:
        envelope = _empty_envelope("unavailable", reason=exc.reason)
        envelope["session_id"] = session_id
        envelope["limit"] = bounded_limit
        return envelope

    if query_result["source_embedded_messages"] == 0:
        envelope = _empty_envelope("not_embedded", reason=None)
        envelope["session_id"] = session_id
        envelope["limit"] = bounded_limit
        return envelope

    results = cast(list[dict[str, object]], query_result["results"])
    hits: list[dict[str, object]] = []
    for hit in results:
        score = float(cast(float, hit["score"]))
        hits.append(
            {
                "session_id": str(hit["session_id"]),
                "score": round(score, 4),
                "distance": round(float(cast(float, hit["distance"])), 4),
                "confidence": _confidence_for_score(score),
                "title": hit["title"],
                "origin": hit["origin"],
                "matched_message_count": int(cast(int, hit["matched_message_count"])),
            }
        )

    # Vector hits that resolve to no indexed message mean the embeddings tier
    # references messages this index does not carry -- a stale embedding
    # generation, or a reindex that changed message-identity derivation. Ranking
    # over a broken join and answering "ready" with the survivors is indis-
    # tinguishable from "nothing is similar", so the caller cannot tell a healthy
    # empty answer from a broken one. Report the discrepancy instead.
    unresolved = int(cast(int, query_result.get("unresolved_message_hits", 0)))
    if unresolved and not hits:
        return {
            "status": "inconsistent",
            "reason": "embedded_messages_missing_from_index",
            "session_id": session_id,
            "source_embedded_messages": int(cast(int, query_result["source_embedded_messages"])),
            "limit": bounded_limit,
            "results": [],
            "unresolved_message_hits": unresolved,
        }

    return {
        "status": "ready",
        "reason": None,
        "session_id": session_id,
        "source_embedded_messages": int(cast(int, query_result["source_embedded_messages"])),
        "limit": bounded_limit,
        "results": hits,
        "unresolved_message_hits": unresolved,
    }


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

    - ``"disabled"`` — embeddings not enabled.
    - ``"unavailable"`` — embeddings are enabled but the ``vec0`` table
      or the ``sqlite-vec`` extension is missing.
    - ``"not_embedded"`` — the source session has no message
      vectors stored yet (waiting on the embedding pipeline).
    - ``"ready"`` — ranked similar sessions are attached under
      ``results``. ``results`` may still be empty if no other
      session shares an embedded neighbor.
    - ``"inconsistent"`` — the vector store returned neighbors but none of
      their message ids exist in this index, so the ranking join is broken
      (a stale embedding generation, or a reindex that changed message
      identity). Distinguished from ``"ready"`` with an empty list, which
      means the archive genuinely holds nothing similar.

    ``unresolved_message_hits`` accompanies ``ready`` and ``inconsistent`` and
    counts neighbors dropped for having no indexed message. Non-zero alongside
    ``ready`` means the ranking is partial.
    """

    bounded_limit = _clamp_limit(limit)
    cfg = load_polylogue_config()
    disabled_reason = _disabled_reason(
        embedding_enabled=bool(cfg.embedding_enabled),
    )

    archive_root_path = archive_root()
    dbf = resolve_active_index_path(archive_root_path)
    if not dbf.exists():
        # Treat a missing database the same as a missing session —
        # the route layer turns this into 404. The reader will never see
        # this branch in practice once the archive has been bootstrapped.
        return None

    return _build_archive_similar_payload(
        str(dbf),
        session_id,
        bounded_limit=bounded_limit,
        disabled_reason=disabled_reason,
        archive_root_path=archive_root_path,
        embedding_model=cfg.embedding_model,
        embedding_dimension=cfg.embedding_dimension,
    )


__all__ = [
    "SIMILAR_RESULTS_DEFAULT",
    "SIMILAR_RESULTS_MAX",
    "build_similar_payload",
]
