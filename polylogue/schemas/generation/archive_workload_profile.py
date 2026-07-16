"""Privacy-safe archive-composition profiles for schema-derived workloads."""

from __future__ import annotations

import gzip
import json
import math
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
from polylogue.schemas.workload_tiers import WorkloadScaleTier, WorkloadSelectivityTier

ARCHIVE_WORKLOAD_PROFILE_FILE = "archive-workload-profile.json.gz"
_INFERENCE_VERSION = "archive-composition-v1"
_CI_SESSION_CEILING = 512
_TAIL_QUANTILES = ("p50", "p95", "p99")


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


def _scan_table_profile(
    conn: sqlite3.Connection,
    table: str,
    *,
    mix_columns: Sequence[str] = (),
    distribution_columns: Sequence[str] = (),
    length_columns: Sequence[str] = (),
) -> tuple[int, JSONDocument, JSONDocument, JSONDocument]:
    """Collect independent row statistics in one base-table pass.

    Length expressions stay inside SQLite so large text/tool payloads are not
    copied into Python merely to measure their encoded size.
    """
    available = _columns(conn, table)
    mixes = [column for column in mix_columns if column in available]
    distributions = [column for column in distribution_columns if column in available]
    lengths = [column for column in length_columns if column in available]
    if not mixes and not distributions and not lengths:
        return _scalar(conn, f'SELECT COUNT(*) FROM "{table}"'), {}, {}, {}

    expressions = [*(f'"{column}"' for column in mixes), *(f'"{column}"' for column in distributions)]
    expressions.extend(f'length(CAST("{column}" AS BLOB))' for column in lengths)
    mix_counts = {column: Counter[str]() for column in mixes}
    sketches = {column: DistributionSketch() for column in distributions}
    distribution_nulls = dict.fromkeys(distributions, 0)
    length_sketches = {column: DistributionSketch() for column in lengths}
    length_nulls = dict.fromkeys(lengths, 0)
    row_count = 0

    for row in conn.execute("SELECT " + ", ".join(expressions) + f' FROM "{table}"'):
        row_count += 1
        offset = 0
        for column in mixes:
            value = row[offset]
            offset += 1
            mix_counts[column]["<null>" if value is None else str(value)] += 1
        for column in distributions:
            value = row[offset]
            offset += 1
            if value is None:
                distribution_nulls[column] += 1
            elif isinstance(value, int | float) and not isinstance(value, bool):
                sketches[column].observe(value)
        for column in lengths:
            value = row[offset]
            offset += 1
            if value is None:
                length_nulls[column] += 1
            elif isinstance(value, int | float) and not isinstance(value, bool):
                length_sketches[column].observe(value)

    distribution_payload: JSONDocument = {}
    for column in distributions:
        payload = sketches[column].to_payload()
        payload["null_count"] = distribution_nulls[column]
        distribution_payload[column] = payload
    length_payload: JSONDocument = {}
    for column in lengths:
        payload = length_sketches[column].to_payload()
        payload["null_count"] = length_nulls[column]
        length_payload[f"{column}_bytes"] = payload
    return (
        row_count,
        {column: dict(sorted(counts.items())) for column, counts in mix_counts.items()},
        distribution_payload,
        length_payload,
    )


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


def _tool_profiles(
    conn: sqlite3.Connection,
    *,
    session_count: int,
) -> tuple[JSONDocument, JSONDocument, JSONDocument]:
    """Derive tool pairing and per-session counts from one grouped scan."""
    required = {"session_id", "tool_id", "block_type"}
    if not required <= _columns(conn, "blocks"):
        return {}, {}, {}
    uses = DistributionSketch()
    results = DistributionSketch()
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
            session_id,
            tool_id,
            SUM(CASE WHEN block_type = 'tool_use' THEN 1 ELSE 0 END) AS uses,
            SUM(CASE WHEN block_type = 'tool_result' THEN 1 ELSE 0 END) AS results
        FROM blocks
        WHERE block_type IN ('tool_use', 'tool_result')
        GROUP BY session_id, tool_id
        ORDER BY session_id, tool_id
    """
    group_size = DistributionSketch()
    current_session: str | None = None
    session_uses = 0
    session_results = 0
    observed_sessions = 0
    for row in conn.execute(query):
        session_id = str(row[0])
        if current_session is not None and session_id != current_session:
            uses.observe(session_uses)
            results.observe(session_results)
            observed_sessions += 1
            session_uses = 0
            session_results = 0
        current_session = session_id
        use_count = int(row[2] or 0)
        result_count = int(row[3] or 0)
        session_uses += use_count
        session_results += result_count
        if row[1] is None or row[1] == "":
            totals["unknown_identity_uses"] += use_count
            totals["unknown_identity_results"] += result_count
            continue
        totals["paired"] += min(use_count, result_count)
        totals["missing_results"] += max(0, use_count - result_count)
        totals["orphan_results"] += max(0, result_count - use_count)
        totals["duplicate_calls"] += max(0, use_count - 1)
        totals["duplicate_results"] += max(0, result_count - 1)
        group_size.observe(use_count + result_count)
    if current_session is not None:
        uses.observe(session_uses)
        results.observe(session_results)
        observed_sessions += 1
    zero_tool_sessions = max(0, session_count - observed_sessions)
    uses.observe_repeated(0, zero_tool_sessions)
    results.observe_repeated(0, zero_tool_sessions)

    use_payload = uses.to_payload()
    use_payload["null_count"] = 0
    result_payload = results.to_payload()
    result_payload["null_count"] = 0
    pairing_payload: JSONDocument = {
        **totals,
        "records_per_tool_identity": group_size.to_payload(),
        "identities_retained": False,
    }
    return pairing_payload, use_payload, result_payload


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
    session_count, session_mixes, session_shapes, session_lengths = _scan_table_profile(
        conn,
        "sessions",
        mix_columns=("origin", "session_kind", "branch_type", "title_source"),
        distribution_columns=(
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
        length_columns=("title", "instructions_text", "git_branch", "git_repository_url", "provider_project_ref"),
    )
    message_count, message_mixes, message_shapes, message_lengths = _scan_table_profile(
        conn,
        "messages",
        mix_columns=("role", "message_type", "material_origin"),
        distribution_columns=(
            "position",
            "variant_index",
            "word_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "duration_ms",
        ),
        length_columns=("user_context_text",),
    )
    block_count, block_mixes, _block_shapes, block_lengths = _scan_table_profile(
        conn,
        "blocks",
        mix_columns=("block_type",),
        length_columns=("text", "tool_input"),
    )
    pairing_profile, tool_use_distribution, tool_result_distribution = _tool_profiles(
        conn,
        session_count=session_count,
    )
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
        "session_shapes": session_shapes,
        "message_shapes": message_shapes,
        "payload_tails": {
            "sessions": session_lengths,
            "messages": message_lengths,
            "blocks": block_lengths,
        },
        "action_shapes": {
            "tool_uses_per_session": tool_use_distribution,
            "tool_results_per_session": tool_result_distribution,
            "tool_pairing": pairing_profile,
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


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _quantile_targets(distribution: object) -> JSONDocument:
    quantiles = _mapping(_mapping(distribution).get("quantiles"))
    return {
        quantile: value
        for quantile in _TAIL_QUANTILES
        if isinstance((value := quantiles.get(quantile)), int | float) and not isinstance(value, bool)
    }


def _shape_anchors(index_profile: Mapping[str, object], source_profile: Mapping[str, object]) -> list[JSONDocument]:
    session_shapes = _mapping(index_profile.get("session_shapes"))
    action_shapes = _mapping(index_profile.get("action_shapes"))
    topology = _mapping(index_profile.get("topology"))
    candidates = (
        ("index.session_shapes.message_count", session_shapes.get("message_count")),
        ("index.action_shapes.tool_uses_per_session", action_shapes.get("tool_uses_per_session")),
        ("index.action_shapes.tool_results_per_session", action_shapes.get("tool_results_per_session")),
        ("index.topology.children_per_parent", topology.get("children_per_parent")),
        ("source.blob_size", source_profile.get("blob_size")),
    )
    anchors: list[JSONDocument] = []
    for distribution_ref, distribution in candidates:
        targets = _quantile_targets(distribution)
        if targets:
            anchors.append({"distribution_ref": distribution_ref, "targets": targets})
    return anchors


def _scaled_row_counts(row_counts: Mapping[str, object], *, target_sessions: int) -> JSONDocument:
    observed_sessions = row_counts.get("sessions")
    if not isinstance(observed_sessions, int) or isinstance(observed_sessions, bool) or observed_sessions <= 0:
        return {}
    scale = target_sessions / observed_sessions
    projected: JSONDocument = {}
    for unit in ("sessions", "messages", "blocks"):
        observed = row_counts.get(unit)
        if isinstance(observed, int) and not isinstance(observed, bool):
            projected[unit] = target_sessions if unit == "sessions" else math.ceil(observed * scale)
    return projected


def _selectivity_targets(index_profile: Mapping[str, object], quantile: str) -> list[JSONDocument]:
    predicate_profile = _mapping(index_profile.get("predicate_selectivity"))
    dimensions = (
        ("repository", "sessions_per_repository"),
        ("branch", "sessions_per_branch"),
        ("model", "messages_per_model"),
        ("tool-name", "blocks_per_tool_name"),
        ("origin", None),
        ("utc-day", None),
    )
    targets: list[JSONDocument] = []
    for dimension, measure in dimensions:
        profile_key = {
            "tool-name": "tool_name",
            "origin": "sessions_per_origin",
            "utc-day": "sessions_per_utc_day",
        }.get(dimension, dimension)
        distribution: object = predicate_profile.get(profile_key)
        if measure is not None:
            distribution = _mapping(distribution).get(measure)
        value = _mapping(_mapping(distribution).get("quantiles")).get(quantile)
        if isinstance(value, int | float) and not isinstance(value, bool):
            targets.append(
                {
                    "dimension": dimension,
                    "distribution_ref": f"index.predicate_selectivity.{profile_key}"
                    + (f".{measure}" if measure is not None else ""),
                    "estimated_cardinality": value,
                }
            )
    return targets


def _named_projection_tiers(
    index_profile: Mapping[str, object],
    source_profile: Mapping[str, object],
) -> tuple[JSONDocument, JSONDocument]:
    row_counts = _mapping(index_profile.get("row_counts"))
    observed_sessions = row_counts.get("sessions")
    session_count = (
        observed_sessions
        if isinstance(observed_sessions, int) and not isinstance(observed_sessions, bool) and observed_sessions > 0
        else 1
    )
    ci_sessions = min(session_count, _CI_SESSION_CEILING)
    shape_anchors = _shape_anchors(index_profile, source_profile)
    archive_1x: JSONDocument = {}
    for unit in ("sessions", "messages", "blocks"):
        value = row_counts.get(unit)
        if isinstance(value, int) and not isinstance(value, bool):
            archive_1x[unit] = value
    archive_10x = {
        unit: value * 10 for unit, value in archive_1x.items() if isinstance(value, int) and not isinstance(value, bool)
    }
    scale_tiers = json_document(
        {
            WorkloadScaleTier.CI_ACTIVATION.value: {
                "purpose": "bounded deterministic projection retaining observed tail and exact-selectivity anchors",
                "row_counts": _scaled_row_counts(row_counts, target_sessions=ci_sessions),
                "session_ceiling": _CI_SESSION_CEILING,
                "shape_anchors": shape_anchors,
                "required_quantiles": list(_TAIL_QUANTILES),
                "selectivity_tier": WorkloadSelectivityTier.EXACT_ONE.value,
            },
            WorkloadScaleTier.ARCHIVE_1X.value: {
                "purpose": "observed archive composition",
                "scale_multiplier": 1,
                "row_counts": archive_1x,
                "shape_anchors": shape_anchors,
            },
            WorkloadScaleTier.ARCHIVE_10X.value: {
                "purpose": "tenfold scaling proof with unchanged distribution contract",
                "scale_multiplier": 10,
                "row_counts": archive_10x,
                "shape_anchors": shape_anchors,
            },
        }
    )
    exact_profile = _mapping(
        _mapping(index_profile.get("predicate_selectivity")).get("exact_existing_session_cardinality")
    )
    exact_cardinality = exact_profile.get("cardinality")
    selectivity_tiers = json_document(
        {
            WorkloadSelectivityTier.EXACT_ONE.value: {
                "purpose": "one exact existing session among archive distractors",
                "dimension": "session-id",
                "cardinality": exact_cardinality if isinstance(exact_cardinality, int) else 1,
                "distribution_ref": "index.predicate_selectivity.exact_existing_session_cardinality",
            },
            WorkloadSelectivityTier.OBSERVED_P50.value: {
                "purpose": "median observed anonymous predicate cardinality",
                "quantile": "p50",
                "targets": _selectivity_targets(index_profile, "p50"),
            },
            WorkloadSelectivityTier.OBSERVED_P99.value: {
                "purpose": "tail observed anonymous predicate cardinality",
                "quantile": "p99",
                "targets": _selectivity_targets(index_profile, "p99"),
            },
        }
    )
    return scale_tiers, selectivity_tiers


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
    scale_tiers, selectivity_tiers = _named_projection_tiers(index_profile, source_profile)
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
        "scale_tiers": scale_tiers,
        "selectivity_tiers": selectivity_tiers,
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
