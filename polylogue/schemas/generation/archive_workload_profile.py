"""Privacy-safe archive-composition profiles for schema-derived workloads."""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path

from polylogue.core.json import JSONDocument, json_document
from polylogue.schemas.field_stats.distributions import DistributionSketch
from polylogue.schemas.generation.workload_profiles import (
    WORKLOAD_PROFILE_VERSION,
    workload_profile_identity,
)

ARCHIVE_WORKLOAD_PROFILE_FILE = "archive-workload-profile.json.gz"
_INFERENCE_VERSION = "archive-composition-v1"


def _connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _scalar(conn: sqlite3.Connection, query: str) -> int:
    row = conn.execute(query).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _mix(conn: sqlite3.Connection, table: str, column: str) -> JSONDocument:
    if column not in _columns(conn, table):
        return {}
    rows = conn.execute(f'SELECT "{column}", COUNT(*) FROM "{table}" GROUP BY "{column}" ORDER BY "{column}"')
    return {"<null>" if row[0] is None else str(row[0]): int(row[1]) for row in rows}


def _mixes(conn: sqlite3.Connection, table: str, columns: Sequence[str]) -> JSONDocument:
    available = _columns(conn, table)
    selected = [column for column in columns if column in available]
    if not selected:
        return {}
    counters = {column: Counter[str]() for column in selected}
    query = "SELECT " + ", ".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    for row in conn.execute(query):
        for index, column in enumerate(selected):
            counters[column]["<null>" if row[index] is None else str(row[index])] += 1
    return {column: dict(sorted(counts.items(), key=lambda item: item[0])) for column, counts in counters.items()}


def _sketch_rows(rows: Iterable[Sequence[object]], index: int = 0) -> JSONDocument:
    sketch = DistributionSketch()
    null_count = 0
    for row in rows:
        value = row[index]
        if value is None:
            null_count += 1
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            sketch.observe(value)
    payload = sketch.to_payload()
    payload["null_count"] = null_count
    return payload


def _column_distributions(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
) -> JSONDocument:
    available = _columns(conn, table)
    selected = [column for column in columns if column in available]
    if not selected:
        return {}
    sketches = {column: DistributionSketch() for column in selected}
    null_counts = dict.fromkeys(selected, 0)
    query = "SELECT " + ", ".join(f'"{column}"' for column in selected) + f' FROM "{table}"'
    for row in conn.execute(query):
        for index, column in enumerate(selected):
            value = row[index]
            if value is None:
                null_counts[column] += 1
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                sketches[column].observe(value)
    payload: JSONDocument = {}
    for column in selected:
        distribution = sketches[column].to_payload()
        distribution["null_count"] = null_counts[column]
        payload[column] = distribution
    return payload


def _length_distributions(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
) -> JSONDocument:
    available = _columns(conn, table)
    selected = [column for column in columns if column in available]
    if not selected:
        return {}
    sketches = {column: DistributionSketch() for column in selected}
    null_counts = dict.fromkeys(selected, 0)
    expressions = ", ".join(f'length(CAST("{column}" AS BLOB))' for column in selected)
    for row in conn.execute(f'SELECT {expressions} FROM "{table}"'):
        for index, column in enumerate(selected):
            value = row[index]
            if value is None:
                null_counts[column] += 1
            elif isinstance(value, int | float) and not isinstance(value, bool):
                sketches[column].observe(value)
    payload: JSONDocument = {}
    for column in selected:
        distribution = sketches[column].to_payload()
        distribution["null_count"] = null_counts[column]
        payload[f"{column}_bytes"] = distribution
    return payload


def _anonymous_cardinality_profile(
    conn: sqlite3.Connection,
    *,
    table: str,
    column: str,
    measure: str = "rows_per_value",
) -> JSONDocument:
    if column not in _columns(conn, table):
        return {}
    sketch = DistributionSketch()
    non_null = 0
    distinct = 0
    for row in conn.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NOT NULL GROUP BY "{column}"'):
        count = int(row[0])
        non_null += count
        distinct += 1
        sketch.observe(count)
    distribution = sketch.to_payload()
    distribution["null_count"] = 0
    return {
        "non_null_observations": non_null,
        "distinct_values": distinct,
        measure: distribution,
        "values_retained": False,
    }


def _per_session_tool_distributions(conn: sqlite3.Connection) -> tuple[JSONDocument, JSONDocument]:
    if not _table_exists(conn, "blocks"):
        return {}, {}
    uses = DistributionSketch()
    results = DistributionSketch()
    rows = conn.execute(
        """
        SELECT
            SUM(CASE WHEN block_type = 'tool_use' THEN 1 ELSE 0 END),
            SUM(CASE WHEN block_type = 'tool_result' THEN 1 ELSE 0 END)
        FROM blocks
        WHERE block_type IN ('tool_use', 'tool_result')
        GROUP BY session_id
        """
    )
    for row in rows:
        use_count = int(row[0] or 0)
        result_count = int(row[1] or 0)
        if use_count:
            uses.observe(use_count)
        if result_count:
            results.observe(result_count)
    use_payload = uses.to_payload()
    use_payload["null_count"] = 0
    result_payload = results.to_payload()
    result_payload["null_count"] = 0
    return use_payload, result_payload


def _tool_pairing_profile(conn: sqlite3.Connection) -> JSONDocument:
    required = {"session_id", "tool_id", "block_type"}
    if not required <= _columns(conn, "blocks"):
        return {}
    totals = {
        "paired": 0,
        "missing_results": 0,
        "orphan_results": 0,
        "duplicate_calls": 0,
        "duplicate_results": 0,
        "unknown_identity_uses": 0,
        "unknown_identity_results": 0,
    }
    query = """
        SELECT
            tool_id,
            SUM(CASE WHEN block_type = 'tool_use' THEN 1 ELSE 0 END) AS uses,
            SUM(CASE WHEN block_type = 'tool_result' THEN 1 ELSE 0 END) AS results
        FROM blocks
        WHERE block_type IN ('tool_use', 'tool_result')
        GROUP BY session_id, tool_id
    """
    group_size = DistributionSketch()
    for row in conn.execute(query):
        uses = int(row[1] or 0)
        results = int(row[2] or 0)
        if row[0] is None or row[0] == "":
            totals["unknown_identity_uses"] += uses
            totals["unknown_identity_results"] += results
            continue
        totals["paired"] += min(uses, results)
        totals["missing_results"] += max(0, uses - results)
        totals["orphan_results"] += max(0, results - uses)
        totals["duplicate_calls"] += max(0, uses - 1)
        totals["duplicate_results"] += max(0, results - 1)
        group_size.observe(uses + results)
    return {**totals, "records_per_tool_identity": group_size.to_payload(), "identities_retained": False}


def _topology_profile(conn: sqlite3.Connection) -> JSONDocument:
    payload: JSONDocument = {}
    if _table_exists(conn, "session_links"):
        payload["link_type_mix"] = _mix(conn, "session_links", "link_type")
        payload["inheritance_mix"] = _mix(conn, "session_links", "inheritance")
        payload["status_mix"] = _mix(conn, "session_links", "status")
        payload["outgoing_links_per_session"] = _sketch_rows(
            conn.execute("SELECT COUNT(*) FROM session_links GROUP BY src_session_id")
        )
        payload["incoming_links_per_session"] = _sketch_rows(
            conn.execute(
                "SELECT COUNT(*) FROM session_links WHERE resolved_dst_session_id IS NOT NULL "
                "GROUP BY resolved_dst_session_id"
            )
        )
    if "parent_session_id" in _columns(conn, "sessions"):
        payload["children_per_parent"] = _sketch_rows(
            conn.execute("SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NOT NULL GROUP BY parent_session_id")
        )
        payload["root_sessions"] = _scalar(conn, "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NULL")
        payload["child_sessions"] = _scalar(conn, "SELECT COUNT(*) FROM sessions WHERE parent_session_id IS NOT NULL")
    return payload


def _index_profile(conn: sqlite3.Connection) -> JSONDocument:
    session_count = _scalar(conn, "SELECT COUNT(*) FROM sessions")
    message_count = _scalar(conn, "SELECT COUNT(*) FROM messages") if _table_exists(conn, "messages") else 0
    block_count = _scalar(conn, "SELECT COUNT(*) FROM blocks") if _table_exists(conn, "blocks") else 0
    session_mixes = _mixes(conn, "sessions", ("origin", "session_kind", "branch_type", "title_source"))
    message_mixes = _mixes(conn, "messages", ("role", "message_type", "material_origin"))
    block_mixes = _mixes(conn, "blocks", ("block_type",))
    tool_use_distribution, tool_result_distribution = _per_session_tool_distributions(conn)
    profile: JSONDocument = {
        "row_counts": {"sessions": session_count, "messages": message_count, "blocks": block_count},
        "closed_vocabulary_mix": {
            "origin": session_mixes.get("origin", {}),
            "session_kind": session_mixes.get("session_kind", {}),
            "branch_type": session_mixes.get("branch_type", {}),
            "title_source": session_mixes.get("title_source", {}),
            "message_role": message_mixes.get("role", {}),
            "message_type": message_mixes.get("message_type", {}),
            "material_origin": message_mixes.get("material_origin", {}),
            "block_type": block_mixes.get("block_type", {}),
        },
        "session_shapes": _column_distributions(
            conn,
            "sessions",
            (
                "message_count",
                "word_count",
                "tool_use_count",
                "thinking_count",
                "paste_count",
                "user_message_count",
                "authored_user_message_count",
                "assistant_message_count",
                "system_message_count",
                "tool_message_count",
                "user_word_count",
                "authored_user_word_count",
                "assistant_word_count",
                "reported_duration_ms",
            ),
        ),
        "message_shapes": _column_distributions(
            conn,
            "messages",
            (
                "position",
                "variant_index",
                "word_count",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "duration_ms",
            ),
        ),
        "payload_tails": {
            "sessions": _length_distributions(
                conn,
                "sessions",
                ("title", "instructions_text", "git_branch", "git_repository_url", "provider_project_ref"),
            ),
            "messages": _length_distributions(conn, "messages", ("user_context_text",)),
            "blocks": _length_distributions(conn, "blocks", ("text", "tool_input")),
        },
        "action_shapes": {
            "tool_uses_per_session": tool_use_distribution,
            "tool_results_per_session": tool_result_distribution,
            "tool_pairing": _tool_pairing_profile(conn),
        },
        "predicate_selectivity": {
            "repository": _anonymous_cardinality_profile(
                conn, table="sessions", column="git_repository_url", measure="sessions_per_repository"
            ),
            "branch": _anonymous_cardinality_profile(
                conn, table="sessions", column="git_branch", measure="sessions_per_branch"
            ),
            "model": _anonymous_cardinality_profile(
                conn, table="messages", column="model_name", measure="messages_per_model"
            ),
            "tool_name": _anonymous_cardinality_profile(
                conn, table="blocks", column="tool_name", measure="blocks_per_tool_name"
            ),
            "sessions_per_origin": _sketch_rows(conn.execute("SELECT COUNT(*) FROM sessions GROUP BY origin")),
            "sessions_per_utc_day": _sketch_rows(
                conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE sort_key_ms IS NOT NULL "
                    "GROUP BY CAST(sort_key_ms / 86400000 AS INTEGER)"
                )
            ),
            "exact_existing_session_cardinality": {
                "count": session_count,
                "cardinality": 1,
            },
        },
        "topology": _topology_profile(conn),
    }
    return profile


def _source_profile(path: Path) -> tuple[JSONDocument, JSONDocument]:
    if not path.is_file():
        return {}, {"source_tier": "source.db missing"}
    with closing(_connect_read_only(path)) as conn:
        if not _table_exists(conn, "raw_sessions"):
            return {}, {"source_tier": "raw_sessions missing"}
        columns = _columns(conn, "raw_sessions")
        window: JSONDocument = {}
        if "acquired_at_ms" in columns:
            row = conn.execute("SELECT MIN(acquired_at_ms), MAX(acquired_at_ms) FROM raw_sessions").fetchone()
            if row is not None:
                window = {"first_acquired_at_ms": row[0], "last_acquired_at_ms": row[1]}
        profile: JSONDocument = {
            "row_count": _scalar(conn, "SELECT COUNT(*) FROM raw_sessions"),
            "origin_mix": _mix(conn, "raw_sessions", "origin"),
            "revision_kind_mix": _mix(conn, "raw_sessions", "revision_kind"),
            "revision_authority_mix": _mix(conn, "raw_sessions", "revision_authority"),
            "capture_mode_mix": _mix(conn, "raw_sessions", "capture_mode"),
            "validation_status_mix": _mix(conn, "raw_sessions", "validation_status"),
            "blob_size": _column_distributions(conn, "raw_sessions", ("blob_size",)).get("blob_size", {}),
            "append_span": _sketch_rows(
                conn.execute(
                    "SELECT append_end_offset - append_start_offset FROM raw_sessions "
                    "WHERE append_start_offset IS NOT NULL AND append_end_offset IS NOT NULL"
                )
            )
            if {"append_start_offset", "append_end_offset"} <= columns
            else {},
            "revisions_per_logical_source": _anonymous_cardinality_profile(
                conn,
                table="raw_sessions",
                column="logical_source_key",
                measure="revisions_per_logical_source",
            ),
            "parse_state": {
                "pending": _scalar(conn, "SELECT COUNT(*) FROM raw_sessions WHERE parsed_at_ms IS NULL")
                if "parsed_at_ms" in columns
                else 0,
                "failed": _scalar(conn, "SELECT COUNT(*) FROM raw_sessions WHERE parse_error IS NOT NULL")
                if "parse_error" in columns
                else 0,
            },
            "observation_window": window,
        }
        generation = (
            _scalar(conn, "SELECT MAX(acquisition_generation) FROM raw_sessions")
            if "acquisition_generation" in columns
            else 0
        )
        return profile, {
            "source_schema_version": int(conn.execute("PRAGMA user_version").fetchone()[0]),
            "max_acquisition_generation": generation,
        }


def _ops_profile(path: Path) -> tuple[JSONDocument, JSONDocument]:
    if not path.is_file():
        return {}, {"ops_tier": "ops.db missing"}
    with closing(_connect_read_only(path)) as conn:
        profile: JSONDocument = {}
        if _table_exists(conn, "ingest_cursor"):
            columns = _columns(conn, "ingest_cursor")
            profile["growing_sources"] = {
                "cursor_count": _scalar(conn, "SELECT COUNT(*) FROM ingest_cursor"),
                "origin_mix": _mix(conn, "ingest_cursor", "origin"),
                "failure_count": _column_distributions(conn, "ingest_cursor", ("failure_count",)).get(
                    "failure_count", {}
                ),
                "record_count": _column_distributions(conn, "ingest_cursor", ("record_count",)).get("record_count", {}),
                "unconsumed_bytes": _sketch_rows(
                    conn.execute("SELECT MAX(COALESCE(stat_size, 0) - COALESCE(byte_offset, 0), 0) FROM ingest_cursor")
                )
                if {"stat_size", "byte_offset"} <= columns
                else {},
                "excluded": _mix(conn, "ingest_cursor", "excluded"),
            }
        if _table_exists(conn, "convergence_debt"):
            profile["convergence_debt"] = {
                "row_count": _scalar(conn, "SELECT COUNT(*) FROM convergence_debt"),
                "stage": _anonymous_cardinality_profile(
                    conn,
                    table="convergence_debt",
                    column="stage",
                    measure="rows_per_stage",
                ),
                "status_mix": _mix(conn, "convergence_debt", "status"),
                "target_type": _anonymous_cardinality_profile(
                    conn,
                    table="convergence_debt",
                    column="target_type",
                    measure="rows_per_target_type",
                ),
                "attempts": _column_distributions(conn, "convergence_debt", ("attempts",)).get("attempts", {}),
            }
        if _table_exists(conn, "cursor_lag_samples"):
            profile["cursor_lag"] = {
                "family": _anonymous_cardinality_profile(
                    conn,
                    table="cursor_lag_samples",
                    column="family",
                    measure="samples_per_family",
                ),
                "severity_mix": _mix(conn, "cursor_lag_samples", "severity"),
                "lag_ms": _column_distributions(conn, "cursor_lag_samples", ("lag_ms",)).get("lag_ms", {}),
                "stuck_file_count": _column_distributions(conn, "cursor_lag_samples", ("stuck_file_count",)).get(
                    "stuck_file_count", {}
                ),
            }
        return profile, {"ops_schema_version": int(conn.execute("PRAGMA user_version").fetchone()[0])}


def _tier_file_sizes(index_path: Path) -> JSONDocument:
    sizes: JSONDocument = {}
    for name in ("source.db", "index.db", "embeddings.db", "user.db", "ops.db"):
        path = index_path.with_name(name)
        if path.is_file():
            sizes[name] = path.stat().st_size
    return sizes


def build_archive_workload_profile(
    index_path: Path,
    *,
    package_bundle_scope_counts: Mapping[str, Mapping[str, int]] | None = None,
    privacy_policy: str = "standard",
) -> JSONDocument | None:
    """Build a content-free profile of archive composition and activation shapes."""
    if not index_path.is_file():
        return None
    with closing(_connect_read_only(index_path)) as conn:
        if not _table_exists(conn, "sessions"):
            return None
        index_profile = _index_profile(conn)
        index_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        window_row = conn.execute(
            "SELECT MIN(created_at_ms), MAX(COALESCE(updated_at_ms, created_at_ms)) FROM sessions"
        ).fetchone()

    source_profile, source_generation = _source_profile(index_path.with_name("source.db"))
    ops_profile, ops_generation = _ops_profile(index_path.with_name("ops.db"))
    loss_inventory: JSONDocument = {}
    loss_inventory.update({key: value for key, value in source_generation.items() if isinstance(value, str)})
    loss_inventory.update({key: value for key, value in ops_generation.items() if isinstance(value, str)})
    archive_generation: JSONDocument = {"index_schema_version": index_version}
    archive_generation.update({key: value for key, value in source_generation.items() if not isinstance(value, str)})
    archive_generation.update({key: value for key, value in ops_generation.items() if not isinstance(value, str)})

    profile: JSONDocument = {
        "profile_version": WORKLOAD_PROFILE_VERSION,
        "profile_kind": "archive-composition",
        "inference_version": _INFERENCE_VERSION,
        "privacy_policy": privacy_policy,
        "privacy_classification": "aggregate-structural-no-raw-content",
        "provenance": {
            "archive_generation": archive_generation,
            "observation_window": {
                "first_session_created_at_ms": window_row[0] if window_row is not None else None,
                "last_session_updated_at_ms": window_row[1] if window_row is not None else None,
                "source": source_profile.get("observation_window", {}),
            },
        },
        "archive_mix": {
            "package_bundle_scope_counts": json_document(
                {provider: dict(versions) for provider, versions in sorted((package_bundle_scope_counts or {}).items())}
            ),
            "tier_file_sizes": _tier_file_sizes(index_path),
        },
        "index": index_profile,
        "source": source_profile,
        "operations": ops_profile,
        "privacy_review": {
            "included_value_dimensions": [
                "closed schema vocabularies",
                "origin and provider package tokens",
            ],
            "suppressed_value_dimensions": [
                "raw content and tool payloads",
                "filesystem paths",
                "repository and branch values",
                "account, project, session, message, and tool identifiers",
                "model and tool names",
                "free-text errors",
                "convergence stage, target-type, and cursor-family values",
            ],
            "suppressed_dimensions_retain_counts_and_anonymous_cardinality": True,
            "potentially_identifying_structural_values_for_operator_review": [
                "observation-window timestamps",
                "origin and provider package/version tokens",
            ],
        },
        "loss_inventory": loss_inventory,
    }
    profile["profile_id"] = workload_profile_identity(profile)
    return profile


def write_archive_workload_profile(output_dir: Path, profile: Mapping[str, object]) -> Path:
    """Write a byte-deterministic gzip artifact for staging or promotion review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ARCHIVE_WORKLOAD_PROFILE_FILE
    payload = json.dumps(dict(profile), sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0))
    return path


__all__ = [
    "ARCHIVE_WORKLOAD_PROFILE_FILE",
    "build_archive_workload_profile",
    "write_archive_workload_profile",
]
