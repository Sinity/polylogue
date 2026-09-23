"""Embedding-status payload builder (substrate, click-free).

Counts come from a sync read connection over the embedding-status tables.
Surfaces (CLI, MCP, dashboards) consume :func:`embedding_status_payload`
and render in their own dialect.
"""

from __future__ import annotations

import json
import shlex
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypedDict

from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.timestamps import iso_from_epoch_ms
from polylogue.storage.embeddings.identity import EmbeddingRecipe
from polylogue.storage.embeddings.materialization import (
    archive_embeddable_message_where,
    archive_embeddable_messages_relation,
    archive_embedding_blocked_counts_sql,
    archive_embedding_messages_table_ref,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.search_providers.sqlite_vec_support import (
    ESTIMATED_TOKENS_PER_MESSAGE,
    VOYAGE_4_COST_PER_1M_TOKENS,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.storage.sqlite.deadline import query_deadline

if TYPE_CHECKING:
    from polylogue.config import Config, PolylogueConfig

DETAIL_QUERY_TIMEOUT_MS = 2_000
DETAIL_CANDIDATE_PROSE_TIMEOUT_MS = 10_000
METADATA_SUMMARY_TIMEOUT_MS = 5_000
STATUS_READ_BUSY_TIMEOUT_MS = 1_000
EMBEDDING_FAILURE_DETAIL_LIMIT = 25


class _HasConfig(Protocol):
    @property
    def config(self) -> Config: ...


class _ArchiveEmbeddingRunRow(Protocol):
    @property
    def run_id(self) -> str: ...

    @property
    def started_at_ms(self) -> int: ...

    @property
    def finished_at_ms(self) -> int | None: ...

    @property
    def status(self) -> str: ...

    @property
    def scanned_sessions(self) -> int: ...

    @property
    def embedded_sessions(self) -> int: ...

    @property
    def skipped_sessions(self) -> int: ...

    @property
    def error_count(self) -> int: ...

    @property
    def embedded_messages(self) -> int: ...

    @property
    def estimated_cost_usd(self) -> float | None: ...

    @property
    def error_message(self) -> str | None: ...


class RetrievalBandPayload(TypedDict, total=False):
    ready: bool
    status: str
    materialized_rows: int
    source_rows: int
    materialized_documents: int
    source_documents: int


class EmbeddingNextActionPayload(TypedDict):
    code: str
    command: str | None
    reason: str


class EmbeddingFailureDetailPayload(TypedDict):
    failure_id: str
    session_id: str
    origin: str
    message_refs: list[str]
    provider: str
    model: str
    error_class: str
    error_message: str
    retryable: bool
    lifecycle_state: str
    created_at: str | None
    updated_at: str | None
    resolution_action: str | None
    supported_actions: list[str]
    resolution_command: str


class EmbeddingCatchupRunPayload(TypedDict):
    run_id: str
    started_at: str
    updated_at: str
    completed_at: str | None
    status: str
    stop_reason: str | None
    rebuild: bool
    max_sessions: int | None
    max_messages: int | None
    stop_after_seconds: int | None
    max_errors: int | None
    planned_sessions: int
    planned_messages: int
    processed_sessions: int
    embedded_sessions: int
    skipped_sessions: int
    error_count: int
    embedded_messages: int
    estimated_cost_usd: float
    last_session_id: str | None


class EmbeddingStatusPayload(TypedDict):
    config_enabled: bool | None
    has_voyage_api_key: bool | None
    daemon_stage_enabled: bool | None
    configured_model: str | None
    configured_dimension: int | None
    monthly_cost_cap_usd: float | None
    status: str
    coverage_measurable: bool
    coverage_unmeasurable_reason: str | None
    total_sessions: int
    embedded_sessions: int | None
    blocked_sessions: int
    embedded_messages: int | None
    pending_sessions: int | None
    pending_messages: int | None
    pending_messages_exact: bool
    candidate_prose_messages: int | None
    candidate_prose_messages_exact: bool
    embedding_coverage_percent: float | None
    embedding_coverage_basis: str
    message_coverage_percent: float | None
    retrieval_ready: bool
    freshness_status: str
    stale_messages: int
    messages_missing_provenance: int
    oldest_embedded_at: str | None
    newest_embedded_at: str | None
    embedding_models: dict[str, int]
    embedding_dimensions: dict[int, int]
    retrieval_bands: dict[str, dict[str, object]]
    failure_count: int
    terminal_failure_count: int
    retryable_failure_count: int
    failure_details: list[EmbeddingFailureDetailPayload]
    total_estimated_cost_usd: float | None
    latest_catchup_run: EmbeddingCatchupRunPayload | None
    latest_material_catchup_run: EmbeddingCatchupRunPayload | None
    next_action: EmbeddingNextActionPayload


@dataclass(frozen=True, slots=True)
class EmbeddingStatusSettings:
    """Explicit configuration facts available to an embedding-status read.

    A legacy :class:`Config` deliberately omits convergence-enable and cost
    settings.  ``None`` therefore means the supplied configuration did not
    observe that fact; it is not a disabled or unbounded default.
    """

    config_enabled: bool | None
    has_voyage_api_key: bool | None
    configured_model: str | None
    configured_dimension: int | None
    monthly_cost_cap_usd: float | None


def embedding_status_settings_from_config(
    config: Config | PolylogueConfig | None,
) -> EmbeddingStatusSettings:
    """Project only configuration facts already supplied by the caller."""

    from polylogue.config import Config, PolylogueConfig

    if config is None:
        return EmbeddingStatusSettings(None, None, None, None, None)
    if isinstance(config, PolylogueConfig):
        return EmbeddingStatusSettings(
            config_enabled=config.embedding_enabled,
            has_voyage_api_key=bool(config.voyage_api_key),
            configured_model=config.embedding_model,
            configured_dimension=config.embedding_dimension,
            monthly_cost_cap_usd=config.embedding_max_cost_usd,
        )
    if isinstance(config, Config):
        index_config = config.index_config
        return EmbeddingStatusSettings(
            config_enabled=None,
            has_voyage_api_key=bool(index_config and index_config.voyage_api_key),
            configured_model=config.embedding_model,
            configured_dimension=config.embedding_dimension,
            monthly_cost_cap_usd=None,
        )
    raise TypeError(f"unsupported embedding status configuration: {type(config).__name__}")


def _payload_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _total_sessions(conn: sqlite3.Connection) -> int:
    from polylogue.storage.embeddings.support import optional_count_sync

    return optional_count_sync(conn, "SELECT COUNT(*) FROM sessions")


def _attached_table_exists(conn: sqlite3.Connection, schema_name: str, table_name: str) -> bool:
    quoted_schema = '"' + schema_name.replace('"', '""') + '"'
    return _table_exists(conn, table_name, schema=quoted_schema)


def _attached_table_name(conn: sqlite3.Connection, schema_name: str, table_name: str) -> str:
    if _attached_table_exists(conn, schema_name, table_name):
        return f"{schema_name}.{table_name}"
    return ""


def _embedding_refs_have_message_semantics(conn: sqlite3.Connection, refs_table: str) -> bool:
    """Return whether a ref relation can bind exact current message semantics."""

    if not refs_table:
        return False
    from polylogue.core.sqlite_introspection import column_exists

    schema, _, table = refs_table.rpartition(".")
    return column_exists(conn, table, "message_content_hash", schema=schema or "main")


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    from polylogue.storage.embeddings.support import is_missing_table_error

    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0
        raise
    if row is None:
        return 0
    return _payload_int(row[0])


def _scalar_int_with_timeout(
    conn: sqlite3.Connection, sql: str, *, timeout_ms: int | None, params: tuple[object, ...] = ()
) -> int | None:
    """Return an exact scalar count, or ``None`` when an owned reader times out.

    A supplied operation reader owns its SQLite progress handler for operation
    cancellation and deadlines. ``timeout_ms=None`` therefore installs no
    competing handler and propagates any outer interruption unchanged.
    """

    from polylogue.storage.embeddings.support import is_missing_table_error

    if timeout_ms is None:
        row = conn.execute(sql, params).fetchone()
        return _payload_int(row[0]) if row else 0

    try:
        with query_deadline(conn, seconds=timeout_ms / 1000):
            row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if is_missing_table_error(exc):
            return 0
        if "interrupted" in message or "locked" in message or "busy" in message:
            return None
        raise
    if row is None:
        return 0
    return _payload_int(row[0])


def _rows_with_timeout(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_ms: int | None,
    params: tuple[object, ...] = (),
) -> list[sqlite3.Row | tuple[object, ...]] | None:
    """Return query rows, or ``None`` when the live archive cannot answer quickly."""

    from polylogue.storage.embeddings.support import is_missing_table_error

    if timeout_ms is None:
        return list(conn.execute(sql, params).fetchall())

    try:
        with query_deadline(conn, seconds=timeout_ms / 1000):
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if is_missing_table_error(exc):
            return []
        if "interrupted" in message or "locked" in message or "busy" in message:
            return None
        raise
    return list(rows)


def _active_failure_details(
    conn: sqlite3.Connection,
    failure_table: str,
    *,
    include_detail: bool,
    timeout_ms: int | None,
) -> list[EmbeddingFailureDetailPayload]:
    """Return bounded active lifecycle rows, never historical acknowledgements."""

    if not include_detail or not failure_table:
        return []
    rows = _rows_with_timeout(
        conn,
        f"""
        SELECT failure_id, session_id, origin, message_refs_json, provider, model, error_class, error_message,
               retryable, lifecycle_state, created_at_ms, updated_at_ms, resolution_action
        FROM {failure_table}
        WHERE lifecycle_state IN ('retryable', 'terminal')
        ORDER BY updated_at_ms DESC, failure_id ASC
        LIMIT ?
        """,
        timeout_ms=timeout_ms,
        params=(EMBEDDING_FAILURE_DETAIL_LIMIT,),
    )
    if rows is None:
        return []
    details: list[EmbeddingFailureDetailPayload] = []
    for row in rows:
        try:
            message_refs = [str(item) for item in json.loads(str(row[3]))]
        except (TypeError, ValueError, json.JSONDecodeError):
            message_refs = []
        details.append(
            {
                "failure_id": str(row[0]),
                "session_id": str(row[1]),
                "origin": str(row[2]),
                "message_refs": message_refs,
                "provider": str(row[4]),
                "model": str(row[5]),
                "error_class": str(row[6]),
                "error_message": str(row[7]),
                "retryable": bool(row[8]),
                "lifecycle_state": str(row[9]),
                "created_at": iso_from_epoch_ms(row[10]),
                "updated_at": iso_from_epoch_ms(row[11]),
                "resolution_action": None if row[12] is None else str(row[12]),
                "supported_actions": ["acknowledge", "requeue", "supersede"],
                "resolution_command": (
                    f"polylogue ops embed resolve-failure {shlex.quote(str(row[0]))}"
                    " --action <acknowledge|requeue|supersede> --yes"
                ),
            }
        )
    return details


def _uniform_embedding_metadata_counts(
    conn: sqlite3.Connection,
    meta_table: str,
    *,
    embedded_messages: int,
    timeout_ms: int | None,
) -> tuple[dict[str, int], dict[int, int]]:
    """Return metadata counts when one model/dimension can be proved quickly."""
    if embedded_messages <= 0:
        return {}, {}
    sample_rows = _rows_with_timeout(
        conn,
        f"""
        SELECT model, dimension
        FROM {meta_table}
        WHERE model IS NOT NULL
          AND dimension IS NOT NULL
        LIMIT 1
        """,
        timeout_ms=timeout_ms,
    )
    if not sample_rows:
        return {}, {}
    model = str(sample_rows[0][0])
    dimension = _payload_int(sample_rows[0][1])
    differing_rows = _rows_with_timeout(
        conn,
        f"""
        SELECT 1
        FROM {meta_table}
        WHERE model IS NULL
           OR dimension IS NULL
           OR model != ?
           OR dimension != ?
        LIMIT 1
        """,
        timeout_ms=timeout_ms,
        params=(model, dimension),
    )
    if differing_rows is None or differing_rows:
        return {}, {}
    return {model: embedded_messages}, {dimension: embedded_messages}


def _sqlite_stat1_index_rows(conn: sqlite3.Connection, index_name: str) -> int | None:
    if not _table_exists(conn, "sqlite_stat1"):
        return None
    try:
        row = conn.execute("SELECT stat FROM sqlite_stat1 WHERE idx = ? LIMIT 1", (index_name,)).fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    first_field = str(row[0]).split(maxsplit=1)[0]
    try:
        return int(first_field)
    except ValueError:
        return None


def _candidate_prose_message_count(
    conn: sqlite3.Connection,
    *,
    timeout_ms: int | None,
) -> tuple[int | None, bool]:
    """Return a bounded count of authored prose candidate rows.

    This deliberately does not join ``blocks`` to enforce the final
    20-character provider-call floor. It answers the cheaper question:
    how many message rows match the paid-embedding prose predicate. The
    payload names this as candidate prose so consumers do not treat it as
    the exact pending-message count.
    """

    stat_rows = _sqlite_stat1_index_rows(conn, "idx_messages_embedding_prose")
    if stat_rows is not None:
        return stat_rows, False

    messages_ref = archive_embedding_messages_table_ref(conn, alias="m")
    exact_count = _scalar_int_with_timeout(
        conn,
        f"""
        SELECT COUNT(*)
        FROM {messages_ref}
        WHERE {archive_embeddable_message_where("m")}
        """,
        timeout_ms=timeout_ms,
    )
    return exact_count, exact_count is not None


def _archive_index_path(db_path: Path) -> Path | None:
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(db_path.parent).active_index_path
    return index_db if index_db.exists() else None


def _coverage_percent(*, embedded_sessions: int, eligible_sessions: int) -> float | None:
    """Return session coverage, or ``None`` when nothing was eligible.

    A zero denominator is a measurement gap, not full coverage: no session was
    weighed, so neither 100.0 nor 0.0 is a fact about this archive.
    """

    if eligible_sessions <= 0:
        return None
    return embedded_sessions / eligible_sessions * 100


def _message_coverage_percent(
    *,
    embedded_messages: int | None,
    candidate_prose_messages: int | None,
    candidate_prose_messages_exact: bool,
) -> float | None:
    if embedded_messages is None:
        return None
    if candidate_prose_messages is None or not candidate_prose_messages_exact:
        return None
    if candidate_prose_messages <= 0:
        return 100.0 if embedded_messages > 0 else 0.0
    return embedded_messages / candidate_prose_messages * 100


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingStateProbe:
    """The outcome of inspecting embedded-vs-pending readiness.

    Three states, deliberately not collapsed:

    ``counts`` present
        Measured. The inspection ran and its four counts are authoritative.
    ``counts`` absent with ``tier_absent`` true
        Measured absence. There is no embeddings tier to inspect, so nothing
        is embedded -- a genuine zero.
    ``counts`` absent with ``tier_absent`` false
        **Unmeasurable.** The inspection could not run (the sqlite-vec
        extension would not load, the refs schema predates the message-level
        contract, or the query timed out). Nothing is known about coverage.

    The third state must never be rendered as the second. ``embeddings.db``
    is the expensive-to-rebuild tier, and reporting an unmeasurable archive as
    ``none`` prescribes a paid regeneration of vectors that may be present and
    intact. ``reason`` retains why the inspection could not certify itself, in
    the same spirit as a retained unknown-outcome reason on a tool result.
    """

    counts: tuple[int, int, int, int] | None
    tier_absent: bool = False
    reason: str | None = None

    @property
    def measurable(self) -> bool:
        return self.counts is not None or self.tier_absent


def _authoritative_archive_embedding_state(
    conn: sqlite3.Connection,
    *,
    refs_table: str,
    meta_table: str,
    vectors_table: str,
    recipe: EmbeddingRecipe,
    timeout_ms: int | None,
) -> ArchiveEmbeddingStateProbe:
    """Count readiness from desired message membership and vector provenance.

    ``embedding_status`` and ``embedding_derivation_state`` are attempt
    telemetry.  They cannot certify a vector: a session is ready only when
    every currently required message has its current ref and a vector meta row
    carrying the complete configured recipe/output contract.  Sessions with no
    embeddable messages are valid-empty partitions.
    """

    if not refs_table and not meta_table and not vectors_table:
        # No embeddings tier at all: a measured absence, not an unknown.
        return ArchiveEmbeddingStateProbe(counts=None, tier_absent=True, reason="embeddings_tier_absent")
    if not refs_table or not meta_table or not vectors_table:
        return ArchiveEmbeddingStateProbe(counts=None, reason="embeddings_tier_incomplete")
    if not _embedding_refs_have_message_semantics(conn, refs_table):
        # A pre-v6 refs schema cannot express the message-level contract this
        # count asserts.  Vectors may well be present; they are unmeasurable.
        return ArchiveEmbeddingStateProbe(counts=None, reason="refs_schema_predates_message_semantics")
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        # ``message_embeddings`` is a vec0 virtual table; without the extension
        # it cannot be read at all.  That is an inability to inspect, never
        # evidence that the vectors are missing.
        return ArchiveEmbeddingStateProbe(
            counts=None,
            reason=f"sqlite_vec_unavailable: {error}" if error is not None else "sqlite_vec_unavailable",
        )
    relation = archive_embeddable_messages_relation(conn, alias="desired", recipe=recipe)
    sql = f"""
        WITH desired_messages AS (
            SELECT message_id, session_id, content_hash, vector_derivation_hash FROM {relation}
        ), per_session AS (
            SELECT d.session_id,
                   COUNT(*) AS required_count,
                   SUM(CASE WHEN r.message_id IS NOT NULL
                              AND r.session_id = d.session_id
                              AND r.message_content_hash = d.content_hash
                              AND r.vector_derivation_hash = d.vector_derivation_hash
                              AND em.recipe_hash = ?
                              AND em.output_contract_hash = ?
                              AND em.model = ?
                              AND em.dimension = ?
                              AND EXISTS(
                                SELECT 1 FROM {vectors_table} AS vectors
                                WHERE vectors.vector_derivation_hash = lower(hex(r.vector_derivation_hash))
                              )
                            THEN 1 ELSE 0 END) AS valid_count
            FROM desired_messages AS d
            LEFT JOIN {refs_table} AS r ON r.message_id = d.message_id
            LEFT JOIN {meta_table} AS em ON em.vector_derivation_hash = r.vector_derivation_hash
            GROUP BY d.session_id
        )
        SELECT
            COALESCE(SUM(CASE WHEN COALESCE(p.required_count, 0) = COALESCE(p.valid_count, 0) THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN COALESCE(p.required_count, 0) > COALESCE(p.valid_count, 0) THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(COALESCE(p.valid_count, 0)), 0),
            COALESCE(SUM(COALESCE(p.required_count, 0) - COALESCE(p.valid_count, 0)), 0)
        FROM sessions AS s
        LEFT JOIN per_session AS p ON p.session_id = s.session_id
        """
    params = (
        recipe.recipe_hash,
        recipe.output_contract_hash,
        recipe.model,
        recipe.dimensions,
    )
    rows = _rows_with_timeout(conn, sql, params=params, timeout_ms=timeout_ms)
    if not rows:
        return ArchiveEmbeddingStateProbe(counts=None, reason="readiness_inspection_timeout")
    counts: tuple[int, int, int, int] = tuple(_payload_int(value) for value in rows[0])  # type: ignore[assignment]
    return ArchiveEmbeddingStateProbe(counts=counts)


def _blocked_archive_embedding_counts(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
    timeout_ms: int | None,
) -> tuple[int, int]:
    """Return (blocked session keys, their still-unembedded required messages).

    Delegates the classification to the embeddings domain; this surface owns
    only the read timeout.  An unavailable or timed-out inspection reports zero
    blocked keys, which leaves them counted as pending -- conservative, never a
    false claim of completeness.
    """

    if not status_table:
        return 0, 0
    sql = archive_embedding_blocked_counts_sql(conn, status_table=status_table, recipe=recipe)
    if sql is None:
        return 0, 0
    from polylogue.storage.embeddings.support import is_missing_table_error

    try:
        rows = _rows_with_timeout(conn, sql, params=(), timeout_ms=timeout_ms)
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0, 0
        raise
    if not rows:
        return 0, 0
    return _payload_int(rows[0][0]), _payload_int(rows[0][1])


def _embedding_status(
    *,
    total_sessions: int,
    embedded_sessions: int,
    pending_sessions: int,
    blocked_sessions: int,
) -> str:
    if total_sessions <= 0:
        return "empty"
    if pending_sessions <= 0 and blocked_sessions <= 0:
        if embedded_sessions <= 0:
            # Sessions exist but none is embedded, pending or blocked: nothing
            # was eligible, so coverage was never measured.  "complete" would
            # advertise a full archive that was never weighed.
            return "unknown"
        return "complete"
    if embedded_sessions <= 0 and blocked_sessions <= 0:
        return "none"
    return "partial"


def _freshness_status(status: str, stats: EmbeddingStatsSnapshot) -> str:
    if stats.embedded_messages is None:
        return status
    if stats.embedded_messages > 0 and (stats.stale_messages > 0 or stats.messages_missing_provenance > 0):
        return "stale"
    return status


def _retrieval_ready(stats: EmbeddingStatsSnapshot) -> bool:
    if stats.embedded_messages is None:
        return False
    return stats.embedded_messages > stats.stale_messages


def _estimated_cost(message_count: int) -> float:
    estimated_tokens = message_count * ESTIMATED_TOKENS_PER_MESSAGE
    return round(estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000, 2)


def _next_action(
    *,
    config_enabled: bool | None,
    has_voyage_api_key: bool | None,
    total_sessions: int,
    embedded_sessions: int,
    pending_sessions: int,
    retrieval_ready: bool,
    stale_messages: int,
    failure_count: int,
    blocked_sessions: int,
) -> EmbeddingNextActionPayload:
    if total_sessions <= 0:
        return {
            "code": "archive_empty",
            "command": None,
            "reason": "Archive contains no sessions to embed.",
        }
    if has_voyage_api_key is None:
        return {
            "code": "settings_not_observed",
            "command": None,
            "reason": "The supplied status configuration does not report whether a Voyage key is available.",
        }
    if not has_voyage_api_key:
        return {
            "code": "set_voyage_key",
            "command": "polylogue ops embed enable --voyage-api-key ...",
            "reason": "Semantic retrieval needs a Voyage API key before embedding can run.",
        }
    if config_enabled is None:
        return {
            "code": "settings_not_observed",
            "command": None,
            "reason": "The supplied status configuration does not report whether embedding convergence is enabled.",
        }
    if failure_count > 0:
        return {
            "code": "inspect_failures",
            "command": "polylogue ops embed status --detail",
            "reason": "Embedding failures exist and need inspection before treating coverage as clean.",
        }
    if blocked_sessions > 0:
        return {
            "code": "acknowledged_terminal_exclusions",
            "command": None,
            "reason": (
                "Some sessions have acknowledged or superseded terminal embedding failures; "
                "they are retained as audit evidence but excluded from automatic retry."
            ),
        }
    if not config_enabled:
        if pending_sessions <= 0 and retrieval_ready:
            return {
                "code": "ready",
                "command": "polylogue --semantic <query>",
                "reason": "Embeddings are retrieval-ready.",
            }
        if embedded_sessions > 0 and pending_sessions > 0:
            return {
                "code": "continue_backfill",
                "command": "polylogue ops embed backfill --yes --max-sessions 10",
                "reason": (
                    "Manual embedding coverage exists, but daemon convergence is disabled; "
                    "continue bounded backfill or enable daemon catch-up."
                ),
            }
        return {
            "code": "enable_embeddings",
            "command": "polylogue ops embed enable --yes",
            "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
        }
    if stale_messages > 0:
        return {
            "code": "refresh_stale",
            "command": "polylogue ops embed backfill --yes --max-sessions 10",
            "reason": "Existing vectors are stale for at least one message.",
        }
    if pending_sessions > 0:
        return {
            "code": "drain_backlog",
            "command": "polylogue ops embed backfill --yes --max-sessions 10",
            "reason": "Embedding convergence is enabled and pending sessions remain.",
        }
    if retrieval_ready:
        return {
            "code": "ready",
            "command": "polylogue --semantic <query>",
            "reason": "Embeddings are retrieval-ready.",
        }
    return {
        "code": "run_preflight",
        "command": "polylogue ops embed preflight --detail",
        "reason": "Embedding state is inconclusive; inspect exact pending-message and retrieval-band details.",
    }


def _unmeasurable_next_action(reason: str | None) -> EmbeddingNextActionPayload:
    """Never prescribe a paid backfill for coverage that was not measured."""

    return {
        "code": "coverage_unmeasurable",
        "command": "polylogue ops embed preflight --detail",
        "reason": (
            "Embedding coverage could not be inspected"
            + (f" ({reason})" if reason else "")
            + "; stored vectors may be present and intact. Restore the inspection before "
            "treating this archive as unembedded -- regenerating vectors is expensive and "
            "an unmeasured archive is not an empty one."
        ),
    }


def _payload_from_stats(
    *,
    settings: EmbeddingStatusSettings,
    total_sessions: int,
    stats: EmbeddingStatsSnapshot,
    latest_catchup_run: EmbeddingCatchupRunPayload | None,
    latest_material_catchup_run: EmbeddingCatchupRunPayload | None,
    pending_messages_exact: bool,
    failure_details: list[EmbeddingFailureDetailPayload] | None = None,
    terminal_failure_count: int = 0,
    retryable_failure_count: int = 0,
    blocked_sessions: int = 0,
    coverage_unmeasurable_reason: str | None = None,
) -> EmbeddingStatusPayload:
    # The snapshot carries its own unmeasurable verdict (the fast route builds
    # no probe), so an unloadable sqlite-vec reaches this boundary as ``None``
    # counts rather than as an explicit argument. Honour either channel.
    coverage_unmeasurable_reason = coverage_unmeasurable_reason or stats.coverage_unmeasurable_reason
    measurable = coverage_unmeasurable_reason is None
    embedded_sessions = stats.embedded_sessions
    pending_sessions = stats.pending_sessions
    eligible_sessions = (
        None
        if embedded_sessions is None or pending_sessions is None
        else embedded_sessions + pending_sessions + blocked_sessions
    )
    if measurable:
        # ``measurable`` is exactly the condition under which the snapshot
        # carries counts; narrow for the type checker.
        assert embedded_sessions is not None and pending_sessions is not None and eligible_sessions is not None
        status = _embedding_status(
            total_sessions=total_sessions,
            embedded_sessions=embedded_sessions,
            pending_sessions=pending_sessions,
            blocked_sessions=blocked_sessions,
        )
        if stats.failure_count > 0 and status == "complete":
            status = "partial"
        retrieval_ready = _retrieval_ready(stats)
    else:
        # The inspection could not run.  Coverage is unknown -- reporting it as
        # a measured absence would prescribe regenerating vectors that may be
        # present and intact.
        status = "unknown"
        retrieval_ready = False
    coverage_percent = (
        _coverage_percent(
            embedded_sessions=embedded_sessions or 0,
            eligible_sessions=eligible_sessions or 0,
        )
        if measurable
        else None
    )
    message_coverage = _message_coverage_percent(
        embedded_messages=stats.embedded_messages,
        candidate_prose_messages=stats.candidate_prose_messages,
        candidate_prose_messages_exact=stats.candidate_prose_messages_exact,
    )
    return {
        "config_enabled": settings.config_enabled,
        "has_voyage_api_key": settings.has_voyage_api_key,
        "daemon_stage_enabled": (
            settings.config_enabled and settings.has_voyage_api_key
            if settings.config_enabled is not None and settings.has_voyage_api_key is not None
            else None
        ),
        "configured_model": settings.configured_model,
        "configured_dimension": settings.configured_dimension,
        "monthly_cost_cap_usd": settings.monthly_cost_cap_usd,
        "status": status,
        "coverage_measurable": measurable,
        "coverage_unmeasurable_reason": coverage_unmeasurable_reason,
        "total_sessions": total_sessions,
        "embedded_sessions": embedded_sessions if measurable else None,
        "blocked_sessions": blocked_sessions,
        "embedded_messages": stats.embedded_messages if measurable else None,
        "pending_sessions": pending_sessions if measurable else None,
        "pending_messages": stats.pending_messages if (measurable and pending_messages_exact) else None,
        "pending_messages_exact": pending_messages_exact and measurable,
        "candidate_prose_messages": stats.candidate_prose_messages,
        "candidate_prose_messages_exact": stats.candidate_prose_messages_exact,
        "embedding_coverage_percent": (round(coverage_percent, 1) if coverage_percent is not None else None),
        "embedding_coverage_basis": "sessions",
        "message_coverage_percent": (
            round(message_coverage, 1) if (measurable and message_coverage is not None) else None
        ),
        "retrieval_ready": retrieval_ready,
        "freshness_status": _freshness_status(status, stats),
        "stale_messages": stats.stale_messages,
        "messages_missing_provenance": stats.messages_missing_provenance,
        "oldest_embedded_at": stats.oldest_embedded_at,
        "newest_embedded_at": stats.newest_embedded_at,
        "embedding_models": stats.model_counts,
        "embedding_dimensions": stats.dimension_counts,
        "retrieval_bands": stats.retrieval_bands,
        "failure_count": stats.failure_count,
        "terminal_failure_count": terminal_failure_count,
        "retryable_failure_count": retryable_failure_count,
        "failure_details": failure_details or [],
        "total_estimated_cost_usd": stats.total_estimated_cost_usd,
        "latest_catchup_run": latest_catchup_run,
        "latest_material_catchup_run": latest_material_catchup_run,
        "next_action": _unmeasurable_next_action(coverage_unmeasurable_reason)
        if not measurable
        else _next_action(
            config_enabled=settings.config_enabled,
            has_voyage_api_key=settings.has_voyage_api_key,
            total_sessions=total_sessions,
            embedded_sessions=embedded_sessions or 0,
            pending_sessions=pending_sessions or 0,
            retrieval_ready=retrieval_ready,
            stale_messages=stats.stale_messages,
            failure_count=stats.failure_count,
            blocked_sessions=blocked_sessions,
        ),
    }


def _archive_embedding_status_payload(
    db_path: Path,
    *,
    settings: EmbeddingStatusSettings,
    include_detail: bool,
    configured_root: Path | None = None,
    _pinned_connection: sqlite3.Connection | None = None,
    _embeddings_schema: str = "embeddings",
    _ops_schema: str = "ops_tier",
) -> EmbeddingStatusPayload | None:
    recipe = EmbeddingRecipe.current(
        model=settings.configured_model or "",
        dimensions=settings.configured_dimension or 0,
    )
    root = configured_root if configured_root is not None else db_path.parent
    owns_connection = _pinned_connection is None
    # Every global readiness probe has a bounded deadline, including an
    # operation-supplied snapshot.  The latter preserves its owner progress
    # handler and receives a thread-safe SQLite interrupt deadline instead.
    detail_timeout_ms = DETAIL_QUERY_TIMEOUT_MS
    metadata_timeout_ms = METADATA_SUMMARY_TIMEOUT_MS
    candidate_prose_timeout_ms = DETAIL_CANDIDATE_PROSE_TIMEOUT_MS
    # Status payloads degrade rather than refuse when the index tier is skewed.
    if _pinned_connection is None:
        index_db = _archive_index_path(db_path)
        if index_db is None:
            return None
        conn = open_readonly_connection(index_db, timeout=STATUS_READ_BUSY_TIMEOUT_MS / 1000.0, validate_schema=False)
    else:
        conn = _pinned_connection
    if owns_connection:
        conn.execute(f"PRAGMA busy_timeout = {STATUS_READ_BUSY_TIMEOUT_MS}")
    # The CLI status fast path may open an active index whose filename is not
    # the conventional ``index.db`` (for example, an active generation).
    # ``open_readonly_connection`` only auto-attaches sibling tiers for the
    # conventional name, so make the two status siblings explicit here.  A
    # missing tier remains a measured absence; an existing attachment is
    # preserved so callers may supply a pinned snapshot.
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
    if owns_connection:
        for schema, filename in (("embeddings", "embeddings.db"), ("ops_tier", "ops.db")):
            sibling = root / filename
            if schema not in aliases and sibling.exists():
                conn.execute(f"ATTACH DATABASE ? AS {schema}", (str(sibling),))
                aliases.add(schema)
    latest_catchup_run: EmbeddingCatchupRunPayload | None = None
    latest_material_catchup_run: EmbeddingCatchupRunPayload | None = None
    try:
        if not _table_exists(conn, "sessions"):
            return None
        if _pinned_connection is not None:
            # Operation snapshots attach only tiers that were available at
            # pin time.  The embeddings tier is optional, so an absent
            # attachment is an explicit unavailable state rather than a
            # relation probe against a nonexistent SQLite schema.  An
            # attached-but-invalid tier still proceeds through the probes
            # below and retains its diagnostic failure.
            aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
            if _embeddings_schema not in aliases:
                return None
            status_table = _attached_table_name(conn, _embeddings_schema, "embedding_status")
            meta_table = _attached_table_name(conn, _embeddings_schema, "message_embeddings_meta")
            failure_table = _attached_table_name(conn, _embeddings_schema, "embedding_failures")
            refs_table = _attached_table_name(conn, _embeddings_schema, "message_embedding_refs")
            vectors_table = _attached_table_name(conn, _embeddings_schema, "message_embeddings")
        elif "embeddings" in aliases:
            status_table = _attached_table_name(conn, "embeddings", "embedding_status")
            meta_table = _attached_table_name(conn, "embeddings", "message_embeddings_meta")
            failure_table = _attached_table_name(conn, "embeddings", "embedding_failures")
            refs_table = _attached_table_name(conn, "embeddings", "message_embedding_refs")
            vectors_table = _attached_table_name(conn, "embeddings", "message_embeddings")
        else:
            status_table = ""
            meta_table = ""
            failure_table = ""
            refs_table = ""
            vectors_table = ""
        has_messages = _table_exists(conn, "messages")
        has_status = bool(status_table)
        has_meta = bool(meta_table)
        has_refs = bool(refs_table)
        has_ref_semantics = _embedding_refs_have_message_semantics(conn, refs_table)
        total_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM sessions")
        authoritative_state = _authoritative_archive_embedding_state(
            conn,
            refs_table=refs_table,
            meta_table=meta_table,
            vectors_table=vectors_table,
            recipe=recipe,
            timeout_ms=detail_timeout_ms if include_detail else metadata_timeout_ms,
        )
        # ``pending_messages_exact`` reports whether the caller paid for an
        # exact backlog count, not whether the embedded-state inspection could
        # certify itself. Those are different questions: when the
        # authoritative state is unavailable nothing is provably embedded, so
        # the detail pass below counts every embeddable message as pending --
        # a conservative backlog that is exact, and consistent with the
        # pending_sessions = total_sessions set in that same branch. The
        # detail pass still downgrades this to False when one of its own
        # queries times out.
        pending_messages_exact = include_detail
        coverage_unmeasurable_reason = None if authoritative_state.measurable else authoritative_state.reason
        if authoritative_state.counts is None:
            # Readiness is unavailable until current refs, recipe metadata,
            # and physical vectors can be inspected together.  Attempt and
            # failure ledgers remain health evidence below, but cannot certify
            # an output that may be absent or stale.  When the tier is simply
            # absent this is a measured zero; otherwise the counts stay
            # unknown and are carried as such all the way to the surface.
            embedded_sessions = 0
            pending_sessions = total_sessions
            embedded_messages = 0
            pending_messages = 0
        else:
            embedded_sessions, pending_sessions, embedded_messages, pending_messages = authoritative_state.counts
        # Pending is `required - valid`, but a key the domain terminally
        # refuses is neither.  The embeddings domain already classifies those
        # keys -- the `blocked` branch of its freshness predicate, the same one
        # that keeps them out of the catchup work set -- so surface that
        # classification rather than counting them as ordinary backlog.
        blocked_sessions, blocked_unembedded_messages = _blocked_archive_embedding_counts(
            conn,
            status_table=status_table,
            recipe=recipe,
            timeout_ms=detail_timeout_ms if include_detail else metadata_timeout_ms,
        )
        # A blocked key was counted as not-valid above, so it is a subset of
        # the pending set; clamping keeps embedded + pending + blocked equal to
        # the eligible session total.
        blocked_sessions = min(blocked_sessions, pending_sessions)
        pending_sessions -= blocked_sessions
        failure_count = (
            _scalar_int(
                conn,
                f"SELECT COUNT(*) FROM {failure_table} WHERE lifecycle_state IN ('retryable', 'terminal')",
            )
            if failure_table
            else (
                _scalar_int(
                    conn,
                    f"""
                SELECT COUNT(*)
                FROM {status_table} AS e
                JOIN sessions AS s ON s.session_id = e.session_id
                WHERE e.error_message IS NOT NULL
                """,
                )
                if has_status
                else 0
            )
        )
        terminal_failure_count = (
            _scalar_int(conn, f"SELECT COUNT(*) FROM {failure_table} WHERE lifecycle_state = 'terminal'")
            if failure_table
            else (
                _scalar_int(
                    conn,
                    f"SELECT COUNT(*) FROM {status_table} WHERE error_message IS NOT NULL AND needs_reindex = 0",
                )
                if has_status
                else 0
            )
        )
        retryable_failure_count = (
            _scalar_int(conn, f"SELECT COUNT(*) FROM {failure_table} WHERE lifecycle_state = 'retryable'")
            if failure_table
            else (
                _scalar_int(
                    conn,
                    f"SELECT COUNT(*) FROM {status_table} WHERE error_message IS NOT NULL AND needs_reindex = 1",
                )
                if has_status
                else 0
            )
        )
        failure_details = _active_failure_details(
            conn,
            failure_table,
            include_detail=include_detail,
            timeout_ms=detail_timeout_ms,
        )
        if authoritative_state.counts is None:
            pending_messages = 0
        candidate_prose_messages: int | None = None
        candidate_prose_messages_exact = False
        stale_messages = 0
        missing_provenance = 0
        oldest_embedded_at: str | None = None
        newest_embedded_at: str | None = None
        model_counts: dict[str, int] = {}
        dimension_counts: dict[int, int] = {}
        if include_detail and has_meta:
            model_rows = _rows_with_timeout(
                conn,
                f"""
                SELECT model, COUNT(*)
                FROM {meta_table}
                GROUP BY model
                ORDER BY COUNT(*) DESC, model ASC
                """,
                timeout_ms=metadata_timeout_ms,
            )
            if model_rows is not None:
                model_counts = {str(row[0]): _payload_int(row[1]) for row in model_rows if row[0] is not None}
            dimension_rows = _rows_with_timeout(
                conn,
                f"""
                SELECT dimension, COUNT(*)
                FROM {meta_table}
                GROUP BY dimension
                ORDER BY COUNT(*) DESC, dimension ASC
                """,
                timeout_ms=metadata_timeout_ms,
            )
            if dimension_rows is not None:
                dimension_counts = {
                    _payload_int(row[0]): _payload_int(row[1]) for row in dimension_rows if row[0] is not None
                }
            if (model_rows is None or dimension_rows is None) and (not model_counts or not dimension_counts):
                fallback_models, fallback_dimensions = _uniform_embedding_metadata_counts(
                    conn,
                    meta_table,
                    embedded_messages=embedded_messages,
                    timeout_ms=metadata_timeout_ms,
                )
                # Only fill the lane that is actually missing: a grouped query
                # that completed carries exact counts and must survive the
                # other lane's timeout.
                if not model_counts:
                    model_counts = fallback_models
                if not dimension_counts:
                    dimension_counts = fallback_dimensions
            bounds_rows = _rows_with_timeout(
                conn,
                f"""
                SELECT MIN(embedded_at_ms), MAX(embedded_at_ms)
                FROM {meta_table}
                """,
                timeout_ms=metadata_timeout_ms,
            )
            if bounds_rows:
                oldest_embedded_at = iso_from_epoch_ms(bounds_rows[0][0])
                newest_embedded_at = iso_from_epoch_ms(bounds_rows[0][1])
        if include_detail and has_messages:
            candidate_prose_messages, candidate_prose_messages_exact = _candidate_prose_message_count(
                conn,
                timeout_ms=candidate_prose_timeout_ms,
            )
            configured_recipe = EmbeddingRecipe.current(
                model=settings.configured_model or "",
                dimensions=settings.configured_dimension or 0,
            )
            messages_ref = archive_embeddable_messages_relation(conn, alias="m", recipe=configured_recipe)
            meta_join = (
                f"LEFT JOIN {meta_table} em ON em.vector_derivation_hash = r.vector_derivation_hash" if has_meta else ""
            )
            vector_present = (
                f"EXISTS(SELECT 1 FROM {vectors_table} AS vectors "
                "WHERE vectors.vector_derivation_hash = lower(hex(r.vector_derivation_hash)))"
                if vectors_table
                else "0"
            )
            total_messages = _scalar_int_with_timeout(
                conn,
                f"SELECT COUNT(*) FROM {messages_ref}",
                timeout_ms=detail_timeout_ms,
            )
            if total_messages is None:
                total_messages = 0
                pending_messages_exact = False
            # v4 (polylogue-q88p): a message is pending unless its ref exists
            # AND that ref's recorded hash matches the message's *current*
            # vector_derivation_hash (computed by the relation above). Presence-
            # based -- there is no per-vector "needs_reindex" anymore.
            if has_refs and embedded_messages == 0:
                pending_messages = total_messages
            elif has_refs and has_ref_semantics and has_meta and vectors_table:
                exact_pending_messages = _scalar_int_with_timeout(
                    conn,
                    f"""
                    SELECT COUNT(*)
                    FROM {messages_ref}
                    LEFT JOIN {refs_table} r ON r.message_id = m.message_id
                    {meta_join}
                    WHERE (
                        r.message_id IS NULL
                        OR r.vector_derivation_hash != m.vector_derivation_hash
                        OR r.message_content_hash IS NOT m.content_hash
                        OR em.vector_derivation_hash IS NULL
                        OR em.recipe_hash != ?
                        OR em.output_contract_hash != ?
                        OR em.model != ?
                        OR em.dimension != ?
                        OR NOT {vector_present}
                      )
                    """,
                    params=(
                        recipe.recipe_hash,
                        recipe.output_contract_hash,
                        recipe.model,
                        recipe.dimensions,
                    ),
                    timeout_ms=detail_timeout_ms,
                )
                if exact_pending_messages is None:
                    pending_messages = 0
                    pending_messages_exact = False
                else:
                    pending_messages = exact_pending_messages
            else:
                pending_messages = total_messages
            if blocked_unembedded_messages:
                # Same basis as the session clamp: these required messages
                # belong to terminally refused keys, so they are blocked, not
                # queued work.
                pending_messages = max(pending_messages - blocked_unembedded_messages, 0)
            if has_refs and embedded_messages == 0:
                missing_provenance = 0
                stale_messages = 0
            elif has_refs and pending_messages_exact:
                if meta_table:
                    exact_missing_provenance = _scalar_int_with_timeout(
                        conn,
                        f"""
                        SELECT COUNT(*)
                        FROM {refs_table} r
                        LEFT JOIN {meta_table} em
                          ON em.vector_derivation_hash = r.vector_derivation_hash
                        WHERE em.vector_derivation_hash IS NULL
                        """,
                        timeout_ms=detail_timeout_ms,
                    )
                    if exact_missing_provenance is None:
                        missing_provenance = 0
                        pending_messages_exact = False
                    else:
                        missing_provenance = exact_missing_provenance
                if pending_messages_exact:
                    exact_stale_messages = _scalar_int_with_timeout(
                        conn,
                        f"""
                        SELECT COUNT(*)
                        FROM {messages_ref}
                        JOIN {refs_table} r ON r.message_id = m.message_id
                        WHERE r.vector_derivation_hash != m.vector_derivation_hash
                        """,
                        timeout_ms=detail_timeout_ms,
                    )
                    if exact_stale_messages is None:
                        stale_messages = 0
                        pending_messages_exact = False
                    else:
                        stale_messages = exact_stale_messages
        stats = EmbeddingStatsSnapshot(
            embedded_sessions=embedded_sessions,
            embedded_messages=embedded_messages,
            pending_sessions=pending_sessions,
            pending_messages=pending_messages,
            candidate_prose_messages=candidate_prose_messages,
            candidate_prose_messages_exact=candidate_prose_messages_exact,
            stale_messages=stale_messages,
            messages_missing_provenance=missing_provenance,
            oldest_embedded_at=oldest_embedded_at,
            newest_embedded_at=newest_embedded_at,
            model_counts=model_counts,
            dimension_counts=dimension_counts,
            retrieval_bands={},
            failure_count=failure_count,
            total_estimated_cost_usd=_estimated_cost(pending_messages)
            if include_detail and pending_messages_exact
            else None,
        )
    finally:
        if owns_connection:
            conn.close()

    latest_catchup_run, latest_material_catchup_run = (
        _archive_catchup_runs_from_connection(conn, schema=_ops_schema)
        if _pinned_connection is not None
        else _archive_catchup_runs(root / "ops.db")
    )

    return _payload_from_stats(
        settings=settings,
        total_sessions=total_sessions,
        stats=stats,
        latest_catchup_run=latest_catchup_run,
        latest_material_catchup_run=latest_material_catchup_run,
        pending_messages_exact=pending_messages_exact,
        failure_details=failure_details,
        terminal_failure_count=terminal_failure_count,
        retryable_failure_count=retryable_failure_count,
        blocked_sessions=blocked_sessions,
        coverage_unmeasurable_reason=coverage_unmeasurable_reason,
    )


def embedding_status_payload_from_connections(
    index_conn: sqlite3.Connection,
    *,
    config: Config | PolylogueConfig | None = None,
    settings: EmbeddingStatusSettings | None = None,
    embeddings_schema: str = "embeddings_tier",
    ops_schema: str = "ops_tier",
    include_detail: bool = False,
) -> EmbeddingStatusPayload | None:
    """Read the canonical embedding payload from pinned attached tiers.

    The caller owns ``index_conn`` and has already forced its snapshot.  This
    adapter shares the normal payload classifier and catchup projection while
    refusing to resolve paths, load config, or create another SQLite handle.
    """

    if embeddings_schema not in {"embeddings", "embeddings_tier"}:
        raise ValueError(f"unsupported embedding status schema: {embeddings_schema!r}")
    if ops_schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported embedding catchup schema: {ops_schema!r}")

    if settings is not None and config is not None:
        raise ValueError("embedding status accepts settings or config, not both")
    return _archive_embedding_status_payload(
        Path("."),
        settings=settings or embedding_status_settings_from_config(config),
        include_detail=include_detail,
        _pinned_connection=index_conn,
        _embeddings_schema=embeddings_schema,
        _ops_schema=ops_schema,
    )


def _run_has_material_signal(run: EmbeddingCatchupRunPayload) -> bool:
    return (
        run["embedded_messages"] > 0
        or run["embedded_sessions"] > 0
        or run["skipped_sessions"] > 0
        or run["error_count"] > 0
        or bool(run["stop_reason"])
    )


def _archive_run_payload(run: _ArchiveEmbeddingRunRow) -> EmbeddingCatchupRunPayload:
    started_at_ms = run.started_at_ms
    finished_at_ms = run.finished_at_ms
    started_at = iso_from_epoch_ms(started_at_ms) or ""
    finished_at = iso_from_epoch_ms(finished_at_ms)
    return {
        "run_id": run.run_id,
        "started_at": started_at,
        "updated_at": finished_at or started_at,
        "completed_at": finished_at,
        "status": run.status,
        "stop_reason": run.error_message,
        "rebuild": False,
        "max_sessions": None,
        "max_messages": None,
        "stop_after_seconds": None,
        "max_errors": None,
        "planned_sessions": run.scanned_sessions,
        "planned_messages": 0,
        "processed_sessions": run.scanned_sessions,
        "embedded_sessions": run.embedded_sessions,
        "skipped_sessions": run.skipped_sessions,
        "error_count": run.error_count,
        "embedded_messages": run.embedded_messages,
        "estimated_cost_usd": float(run.estimated_cost_usd or 0.0),
        "last_session_id": None,
    }


def _archive_catchup_runs(
    ops_db: Path,
) -> tuple[EmbeddingCatchupRunPayload | None, EmbeddingCatchupRunPayload | None]:
    if not ops_db.exists():
        return None, None
    try:
        conn = open_readonly_connection(ops_db)
        try:
            return _archive_catchup_runs_from_connection(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        return None, None


def _archive_catchup_runs_from_connection(
    conn: sqlite3.Connection,
    *,
    schema: str = "main",
) -> tuple[EmbeddingCatchupRunPayload | None, EmbeddingCatchupRunPayload | None]:
    """Read canonical catchup history from an already-pinned ops attachment."""

    from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported embedding catchup reader schema: {schema!r}")
    aliases = {str(row[1]) for row in conn.execute("PRAGMA database_list").fetchall()}
    if schema not in aliases:
        return None, None
    table = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_schema WHERE type = 'table' AND name = 'embedding_catchup_runs'"
    ).fetchone()
    if table is None:
        return None, None
    runs = list_embedding_catchup_runs(conn, schema=schema)
    if not runs:
        return None, None
    latest = _archive_run_payload(runs[0])
    latest_material = next(
        (payload for payload in (_archive_run_payload(run) for run in runs) if _run_has_material_signal(payload)),
        None,
    )
    return latest, latest_material


def embedding_status_payload(
    env: _HasConfig,
    *,
    include_retrieval_bands: bool = False,
    include_detail: bool = False,
) -> EmbeddingStatusPayload:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.config import load_polylogue_config
    from polylogue.storage.archive_identity import archive_file_set_root

    cfg = load_polylogue_config()
    db_path = Path(env.config.db_path)
    # `env.config` is a duck-typed `_HasConfig` seam (see embedding_readiness_info,
    # which passes a bare SimpleNamespace(db_path=...) with no `archive_root`),
    # not always a real Config -- fall back to the plain db_path.parent
    # derivation archive_file_set_root itself uses when archive_root is
    # unavailable, matching this seam's original (pre-split-root-aware) behavior.
    archive_root = getattr(env.config, "archive_root", None)
    configured_root = (
        archive_file_set_root(archive_root=archive_root, db_path=db_path)
        if archive_root is not None
        else db_path.parent
    )
    settings = embedding_status_settings_from_config(cfg)
    archive_payload = _archive_embedding_status_payload(
        db_path,
        settings=settings,
        include_detail=include_detail,
        configured_root=configured_root,
    )
    if archive_payload is not None:
        return archive_payload
    # The split-file archive is the sole runtime. An index tier that yields no
    # archive-shaped embedding state has nothing to measure; there is no
    # pre-split single-file shape to fall back to.
    return _payload_from_stats(
        settings=settings,
        total_sessions=0,
        stats=EmbeddingStatsSnapshot(),
        latest_catchup_run=None,
        latest_material_catchup_run=None,
        pending_messages_exact=include_detail,
    )
