"""Build a focused claim-vs-evidence report from structured failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from polylogue.config import Config, get_config
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from scripts.agent_forensics import _classify_failed_followup_evidence

_WORDLESS_CONTINUATION_TEXT_CHAR_LIMIT = 40
_COUNT_KEYS = (
    "failed_outcomes",
    "acknowledged",
    "silent_proceed",
    "ambiguous",
    "ambiguous_wordless_continuation",
    "ambiguous_prose_no_marker",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace claim-vs-evidence",
        description="Build a focused report over structured tool failures and the next assistant turn.",
    )
    parser.add_argument("--archive-root", type=Path, default=None, help="Override the active archive root.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write report.json, summary.json, and README.md.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum structured failed outcomes to classify.")
    parser.add_argument("--sample-limit", type=int, default=30, help="Maximum evidence samples per class.")
    parser.add_argument("--json", action="store_true", help="Emit JSON report to stdout.")
    return parser


def _config_with_archive_root(config: Config, archive_root: Path | None) -> Config:
    if archive_root is None:
        return config
    resolved = archive_root.expanduser().resolve()
    return Config(
        archive_root=resolved,
        render_root=config.render_root,
        sources=config.sources,
        db_path=resolved / "index.db",
        drive_config=config.drive_config,
        index_config=config.index_config,
    )


def _user_version(conn: Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _rows(conn: Connection, sql: str, params: Iterable[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    columns = [str(description[0]) for description in cursor.description or ()]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _scalar_int(conn: Connection, sql: str, params: Iterable[object] = ()) -> int:
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def _object_int(value: object) -> int:
    if value is None:
        return 0
    return int(str(value))


def _ranked(mapping: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, counts in mapping.items():
        failed = counts["failed_outcomes"]
        silent = counts["silent_proceed"]
        classified = counts["acknowledged"] + silent
        rows.append(
            {
                "name": name,
                **counts,
                "classified_outcomes": classified,
                "silent_rate_lower_bound": (silent / failed) if failed else 0.0,
                "silent_rate_among_classified": (silent / classified) if classified else 0.0,
            }
        )

    def sort_key(row: dict[str, object]) -> tuple[int, str]:
        failed = row["failed_outcomes"]
        failed_count = failed if isinstance(failed, int) else int(str(failed))
        return (-failed_count, str(row["name"]))

    return sorted(rows, key=sort_key)


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _refine_classification_reason(classification_evidence: Mapping[str, object], row: dict[str, object]) -> str:
    reason = str(classification_evidence["reason"])
    if classification_evidence["classification"] != "ambiguous":
        return reason
    if reason == "missing_next_assistant_message":
        return reason
    has_tool_use = bool(_object_int(row["next_has_tool_use"]))
    pre_tool_text_chars = _object_int(row["next_pre_tool_text_chars"])
    if has_tool_use and pre_tool_text_chars <= _WORDLESS_CONTINUATION_TEXT_CHAR_LIMIT:
        return "wordless_tool_continuation"
    return "prose_no_marker"


def _ambiguous_counter_key(classification_reason: str) -> str | None:
    if classification_reason == "wordless_tool_continuation":
        return "ambiguous_wordless_continuation"
    if classification_reason == "prose_no_marker":
        return "ambiguous_prose_no_marker"
    return None


def _next_message_details(
    conn: Connection, message_ids: Iterable[object], *, chunk_size: int = 500
) -> dict[str, dict[str, object]]:
    ids = [str(message_id) for message_id in message_ids if message_id]
    details: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = _rows(
            conn,
            f"""
            SELECT
                message_id,
                position,
                block_type,
                COALESCE(text, '') AS text
            FROM blocks
            WHERE message_id IN ({placeholders})
            ORDER BY message_id, position
            """,
            chunk,
        )
        for row in rows:
            message_id = str(row["message_id"])
            detail = details.setdefault(
                message_id,
                {
                    "next_has_tool_use": 0,
                    "first_tool_use_position": None,
                    "next_pre_tool_text_chars": 0,
                    "text_parts": [],
                },
            )
            block_type = str(row["block_type"])
            position = _object_int(row["position"])
            if block_type == "tool_use":
                detail["next_has_tool_use"] = 1
                first_tool_use_position = detail["first_tool_use_position"]
                if first_tool_use_position is None or position < _object_int(first_tool_use_position):
                    detail["first_tool_use_position"] = position
            elif block_type == "text":
                text = str(row["text"] or "")
                text_parts = detail["text_parts"]
                assert isinstance(text_parts, list)
                text_parts.append(text)
                first_tool_use_position = detail["first_tool_use_position"]
                if first_tool_use_position is None or position < _object_int(first_tool_use_position):
                    detail["next_pre_tool_text_chars"] = max(
                        _object_int(detail["next_pre_tool_text_chars"]),
                        len(text.strip()),
                    )
    return {
        message_id: {
            "next_text": "\n".join(str(part) for part in detail["text_parts"])[:1200],
            "next_has_tool_use": _object_int(detail["next_has_tool_use"]),
            "next_pre_tool_text_chars": _object_int(detail["next_pre_tool_text_chars"]),
        }
        for message_id, detail in details.items()
    }


def _failure_outcome_rows(conn: Connection, *, limit: int, origin: str | None) -> list[dict[str, object]]:
    origin_predicate = "AND s.origin = ?" if origin is not None else ""
    params: tuple[object, ...] = (origin, origin, origin, limit) if origin is not None else (limit,)
    return _rows(
        conn,
        f"""
        WITH failed AS (
            SELECT
                r.session_id,
                r.message_id AS tool_result_message_id,
                r.tool_id AS tool_result_tool_id,
                s.origin,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                r.message_id AS order_message_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN sessions AS s ON s.session_id = r.session_id
            WHERE r.block_type = 'tool_result'
              {origin_predicate}
              AND r.tool_result_is_error = 1
            UNION ALL
            SELECT
                r.session_id,
                r.message_id AS tool_result_message_id,
                r.tool_id AS tool_result_tool_id,
                s.origin,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                r.message_id AS order_message_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN sessions AS s ON s.session_id = r.session_id
            WHERE r.block_type = 'tool_result'
              {origin_predicate}
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error = 0
            UNION ALL
            SELECT
                r.session_id,
                r.message_id AS tool_result_message_id,
                r.tool_id AS tool_result_tool_id,
                s.origin,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                r.message_id AS order_message_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN sessions AS s ON s.session_id = r.session_id
            WHERE r.block_type = 'tool_result'
              {origin_predicate}
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error IS NULL
        )
        SELECT *
        FROM failed
        ORDER BY session_id, tool_result_tool_id, order_message_id
        LIMIT ?
        """,
        params,
    )


def _paired_failure_rows(
    conn: Connection,
    failure_rows: list[dict[str, object]],
    *,
    chunk_size: int = 250,
) -> list[dict[str, object]]:
    paired_rows: list[dict[str, object]] = []
    for start in range(0, len(failure_rows), chunk_size):
        chunk = failure_rows[start : start + chunk_size]
        placeholders = ",".join("(?, ?, ?, ?, ?, ?, ?, ?)" for _ in chunk)
        params: list[object] = []
        for offset, row in enumerate(chunk, start=start):
            params.extend(
                [
                    offset,
                    row["session_id"],
                    row["tool_result_message_id"],
                    row["tool_result_tool_id"],
                    row["origin"],
                    row["is_error"],
                    row["exit_code"],
                    row["order_message_id"],
                ]
            )
        paired_rows.extend(
            _rows(
                conn,
                f"""
                WITH wanted(
                    sort_index,
                    session_id,
                    tool_result_message_id,
                    tool_result_tool_id,
                    origin,
                    is_error,
                    exit_code,
                    order_message_id
                ) AS (
                    VALUES {placeholders}
                ),
                paired AS (
                    SELECT
                        w.sort_index,
                        w.session_id,
                        u.message_id,
                        w.tool_result_message_id,
                        w.tool_result_tool_id,
                        u.tool_name,
                        u.tool_command,
                        w.origin,
                        w.is_error,
                        w.exit_code,
                        m.model_name AS tool_message_model,
                        (
                            SELECT nm.message_id
                            FROM messages AS nm
                            JOIN messages AS rm ON rm.message_id = w.tool_result_message_id
                            WHERE nm.session_id = w.session_id
                              AND nm.role = 'assistant'
                              AND nm.position > rm.position
                            ORDER BY nm.position
                            LIMIT 1
                        ) AS next_message_id
                    FROM wanted AS w
                    JOIN blocks AS u INDEXED BY idx_blocks_tool_id
                      ON u.tool_id = w.tool_result_tool_id
                     AND u.session_id = w.session_id
                     AND u.block_type = 'tool_use'
                    JOIN messages AS m ON m.message_id = u.message_id
                )
                SELECT
                    p.session_id,
                    p.message_id,
                    p.tool_result_message_id,
                    p.tool_result_tool_id,
                    p.tool_name,
                    p.tool_command,
                    p.origin,
                    p.is_error,
                    p.exit_code,
                    COALESCE(nm.model_name, p.tool_message_model, '') AS model_name,
                    p.next_message_id
                FROM paired AS p
                LEFT JOIN messages AS nm ON nm.message_id = p.next_message_id
                ORDER BY p.sort_index
                """,
                params,
            )
        )
    return paired_rows


def _structured_failure_rows(conn: Connection, *, limit: int, origin: str | None = None) -> list[dict[str, object]]:
    candidate_limit = max(limit * 2, limit + 100)
    rows = _paired_failure_rows(conn, _failure_outcome_rows(conn, limit=candidate_limit, origin=origin))[:limit]
    details = _next_message_details(conn, (row.get("next_message_id") for row in rows))
    for row in rows:
        detail = details.get(str(row.get("next_message_id") or ""), {})
        row["next_text"] = detail.get("next_text", "")
        row["next_has_tool_use"] = detail.get("next_has_tool_use", 0)
        row["next_pre_tool_text_chars"] = detail.get("next_pre_tool_text_chars", 0)
    return rows


def _unpaired_structured_failure_count(conn: Connection) -> int:
    return _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM (
            SELECT r.session_id, r.tool_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_is_error = 1
            UNION ALL
            SELECT r.session_id, r.tool_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error = 0
            UNION ALL
            SELECT r.session_id, r.tool_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error IS NULL
        ) AS r
        WHERE NOT EXISTS (
              SELECT 1
              FROM blocks AS u INDEXED BY idx_blocks_tool_id
              WHERE u.tool_id = r.tool_id
                AND u.session_id = r.session_id
                AND u.block_type = 'tool_use'
          )
        """,
    )


def _structured_failure_origin_counts(conn: Connection) -> list[dict[str, object]]:
    return _rows(
        conn,
        """
        WITH failed AS (
            SELECT r.session_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_is_error = 1
            UNION ALL
            SELECT r.session_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error = 0
            UNION ALL
            SELECT r.session_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            WHERE r.block_type = 'tool_result'
              AND r.tool_result_exit_code IS NOT NULL
              AND r.tool_result_exit_code != 0
              AND r.tool_result_is_error IS NULL
        )
        SELECT s.origin, COUNT(*) AS failed_outcomes
        FROM failed AS r
        JOIN sessions AS s ON s.session_id = r.session_id
        GROUP BY s.origin
        ORDER BY failed_outcomes DESC, s.origin
        """,
    )


def _origin_sample_limits(total_by_origin: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    origins = [(str(row["origin"]), _object_int(row["failed_outcomes"])) for row in total_by_origin]
    origins = [(origin, count) for origin, count in origins if count > 0]
    total = sum(count for _, count in origins)
    if total <= limit:
        return [
            {"origin": origin, "total_structured_failures": count, "requested_limit": count}
            for origin, count in origins
        ]
    if not origins:
        return []
    if limit < len(origins):
        return [
            {"origin": origin, "total_structured_failures": count, "requested_limit": 1}
            for origin, count in origins[:limit]
        ]

    allocation = {origin: 1 for origin, _ in origins}
    remaining = limit - len(origins)
    capacities = {origin: count - 1 for origin, count in origins}
    capacity_total = sum(capacities.values())
    remainders: list[tuple[float, int, str]] = []
    if capacity_total:
        for index, (origin, _count) in enumerate(origins):
            exact = remaining * (capacities[origin] / capacity_total)
            extra = min(capacities[origin], int(exact))
            allocation[origin] += extra
            remainders.append((exact - extra, -index, origin))
        assigned = sum(allocation.values())
        for _remainder, _negative_index, origin in sorted(remainders, reverse=True):
            if assigned >= limit:
                break
            if allocation[origin] < dict(origins)[origin]:
                allocation[origin] += 1
                assigned += 1

    return [
        {"origin": origin, "total_structured_failures": count, "requested_limit": allocation[origin]}
        for origin, count in origins
        if allocation[origin] > 0
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")
    config = _config_with_archive_root(get_config(), args.archive_root)
    index_db = config.db_path
    conn = open_readonly_connection(index_db)
    try:
        total_by_origin = _structured_failure_origin_counts(conn)
        total_structured_failures = sum(_object_int(row["failed_outcomes"]) for row in total_by_origin)
        unpaired_structured_failures = _unpaired_structured_failure_count(conn)
        origin_limits = _origin_sample_limits(total_by_origin, args.limit)
        rows = []
        sampled_by_origin: list[dict[str, object]] = []
        for origin_limit in origin_limits:
            origin = str(origin_limit["origin"])
            requested_limit = _object_int(origin_limit["requested_limit"])
            origin_rows = _structured_failure_rows(conn, limit=requested_limit, origin=origin)
            rows.extend(origin_rows)
            sampled_by_origin.append(
                {
                    **origin_limit,
                    "inspected_structured_failures": len(origin_rows),
                }
            )
        schema_version = _user_version(conn)
    finally:
        conn.close()

    totals = _empty_counts()
    by_tool: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    by_origin: dict[str, dict[str, int]] = {}
    samples_by_classification: dict[str, list[dict[str, object]]] = {
        "acknowledged": [],
        "silent_proceed": [],
        "ambiguous": [],
    }
    samples_by_origin_classification: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        classification_evidence = _classify_failed_followup_evidence(
            str(row["next_text"]) if row["next_text"] is not None else None
        )
        classification = str(classification_evidence["classification"])
        classification_reason = _refine_classification_reason(classification_evidence, row)
        tool = str(row["tool_name"] or "unknown")
        model = str(row["model_name"] or "unknown")
        origin = str(row["origin"] or "unknown")
        next_text = str(row["next_text"] or "")
        sample = {
            "classification": classification,
            "classification_reason": classification_reason,
            "matched_marker": classification_evidence["matched_marker"],
            "session_ref": f"session:{row['session_id']}",
            "tool_message_ref": f"message:{row['message_id']}",
            "tool_result_message_ref": f"message:{row['tool_result_message_id']}",
            "tool_result_tool_id": row["tool_result_tool_id"],
            "next_message_ref": f"message:{row['next_message_id']}" if row["next_message_id"] else None,
            "tool_name": tool,
            "model_name": model,
            "origin": origin,
            "exit_code": row["exit_code"],
            "is_error": row["is_error"],
            "tool_command_preview": str(row["tool_command"] or "")[:160],
            "next_text_preview": next_text[:500],
            "next_has_tool_use": bool(_object_int(row["next_has_tool_use"])),
            "next_pre_tool_text_chars": _object_int(row["next_pre_tool_text_chars"]),
        }
        totals["failed_outcomes"] += 1
        totals[classification] += 1
        ambiguous_counter_key = _ambiguous_counter_key(classification_reason)
        if ambiguous_counter_key is not None:
            totals[ambiguous_counter_key] += 1
        by_tool.setdefault(tool, _empty_counts())
        by_model.setdefault(model, _empty_counts())
        by_origin.setdefault(origin, _empty_counts())
        by_tool[tool]["failed_outcomes"] += 1
        by_tool[tool][classification] += 1
        if ambiguous_counter_key is not None:
            by_tool[tool][ambiguous_counter_key] += 1
        by_model[model]["failed_outcomes"] += 1
        by_model[model][classification] += 1
        if ambiguous_counter_key is not None:
            by_model[model][ambiguous_counter_key] += 1
        by_origin[origin]["failed_outcomes"] += 1
        by_origin[origin][classification] += 1
        if ambiguous_counter_key is not None:
            by_origin[origin][ambiguous_counter_key] += 1
        bucket = samples_by_classification[classification]
        if len(bucket) < args.sample_limit:
            bucket.append(sample)
        origin_buckets = samples_by_origin_classification.setdefault(
            origin,
            {
                "acknowledged": [],
                "silent_proceed": [],
                "ambiguous": [],
            },
        )
        origin_bucket = origin_buckets[classification]
        if len(origin_bucket) < args.sample_limit:
            origin_bucket.append(sample)
    totals["classified_outcomes"] = totals["acknowledged"] + totals["silent_proceed"]
    silent = totals["silent_proceed"]
    failed = totals["failed_outcomes"]
    classified = totals["classified_outcomes"]
    report: dict[str, Any] = {
        "report_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "command": "devtools workspace claim-vs-evidence",
        "archive_root": str(config.archive_root),
        "index_db": str(index_db),
        "index_schema_version": schema_version,
        "limit": args.limit,
        "sample_frame": {
            "total_structured_failures": total_structured_failures,
            "unpaired_structured_failures": unpaired_structured_failures,
            "inspected_structured_failures": len(rows),
            "limit": args.limit,
            "time_window": "entire archive (no since/until filter)",
            "complete_failure_frame": len(rows) >= total_structured_failures,
            "selection_strategy": (
                "origin-stratified bounded sample; at least one row per origin when limit allows, "
                "then proportional fill by origin failure count; each origin candidate frame is bounded "
                "before pairing to tool-use rows"
            ),
            "selection_order": "origin, session_id, tool_id, tool_result_message_id",
            "failure_predicate": "tool_result_is_error = 1 OR tool_result_exit_code != 0",
            "classification_scope": "immediately following assistant message only",
            "total_by_origin": total_by_origin,
            "sampled_by_origin": sampled_by_origin,
        },
        "definition": (
            "Structured failures are normalized tool-result outcomes with is_error=1 or non-zero exit_code. "
            "The immediately following assistant message is classified only for explicit failure "
            "acknowledgment markers; this is not an LLM judgment or prose-mined outcome."
        ),
        "totals": totals,
        "rates": {
            "silent_rate_lower_bound": (silent / failed) if failed else 0.0,
            "silent_rate_among_classified": (silent / classified) if classified else 0.0,
        },
        "by_tool": _ranked(by_tool),
        "by_model": _ranked(by_model),
        "by_origin": _ranked(by_origin),
        "samples_by_classification": samples_by_classification,
        "samples_by_origin_classification": samples_by_origin_classification,
    }
    if args.out_dir is not None:
        _write_artifacts(args.out_dir, report)
    return report


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_artifacts(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "claim-vs-evidence.report.json", report)
    totals = report["totals"]
    rates = report["rates"]
    summary = {
        "artifact": "claim-vs-evidence",
        "updated_at": report["captured_at"],
        "archive_root": report["archive_root"],
        "index_schema_version": report["index_schema_version"],
        "claim": (
            "Polylogue can produce a bounded claim-vs-evidence report by anchoring on structured "
            "tool failures and classifying the immediately following assistant turn for explicit "
            "failure acknowledgment."
        ),
        "non_claim": (
            "This is not a whole-archive rate unless --limit exceeds all structured failures, and it is "
            "not an LLM judgment of intent or utility. Ambiguous follow-ups remain in the denominator."
        ),
        "proof_report": {
            "failed_outcomes": totals["failed_outcomes"],
            "total_structured_failures": report["sample_frame"]["total_structured_failures"],
            "unpaired_structured_failures": report["sample_frame"]["unpaired_structured_failures"],
            "complete_failure_frame": report["sample_frame"]["complete_failure_frame"],
            "acknowledged": totals["acknowledged"],
            "silent_proceed": totals["silent_proceed"],
            "ambiguous": totals["ambiguous"],
            "ambiguous_wordless_continuation": totals["ambiguous_wordless_continuation"],
            "ambiguous_prose_no_marker": totals["ambiguous_prose_no_marker"],
            "silent_rate_lower_bound": rates["silent_rate_lower_bound"],
            "limit": report["limit"],
            "time_window": report["sample_frame"]["time_window"],
            "sampled_by_origin": report["sample_frame"]["sampled_by_origin"],
        },
        "caveats": [
            "The report is bounded by --limit for fast active-archive regeneration.",
            "Classification inspects only the next assistant message for explicit acknowledgment markers.",
            "Structured failure truth comes from normalized action result is_error/exit_code fields, not assistant prose.",
            "Failed tool results without a paired tool-use row are reported as unpaired coverage gaps, not classified rows.",
        ],
        "source_files": ["claim-vs-evidence.report.json"],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_readme(out_dir / "README.md", report)


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    totals = report["totals"]
    rates = report["rates"]
    frame = report["sample_frame"]
    sampled_by_origin = [
        (
            f"- {row['origin']}: inspected {int(row['inspected_structured_failures']):,} / "
            f"{int(row['total_structured_failures']):,} structured failures "
            f"(requested {int(row['requested_limit']):,})"
        )
        for row in frame["sampled_by_origin"]
    ]
    lines = [
        "# Claim-vs-Evidence Failure Follow-Up",
        "",
        f"Generated: {report['captured_at']}",
        f"Archive root: `{report['archive_root']}`",
        f"Index schema: v{report['index_schema_version']}",
        "",
        "## What This Proves",
        "",
        "This demo anchors on structured tool-result evidence and asks what the next assistant",
        "turn did with that failure. It does not infer truth from assistant prose: the failure",
        "predicate is `is_error=1` or a non-zero `exit_code` on normalized `actions` rows.",
        "",
        "## Current Bounded Result",
        "",
        f"- time window: {frame['time_window']}",
        f"- total structured failures in frame: {frame['total_structured_failures']:,}",
        f"- unpaired structured failures outside classifiable frame: {frame['unpaired_structured_failures']:,}",
        f"- failed structured outcomes inspected: {totals['failed_outcomes']:,}",
        f"- complete failure frame: {frame['complete_failure_frame']}",
        f"- acknowledged: {totals['acknowledged']:,}",
        f"- silent-proceed: {totals['silent_proceed']:,}",
        f"- ambiguous: {totals['ambiguous']:,}",
        f"- ambiguous wordless tool continuations: {totals['ambiguous_wordless_continuation']:,}",
        f"- ambiguous prose without markers: {totals['ambiguous_prose_no_marker']:,}",
        f"- silent lower bound: {rates['silent_rate_lower_bound']:.1%}",
        f"- silent among classified: {rates['silent_rate_among_classified']:.1%}",
        f"- configured limit: {report['limit']:,}",
        f"- selection order: {frame['selection_order']}",
        f"- selection strategy: {frame['selection_strategy']}",
        "",
        "### Inspected vs Total by Origin",
        "",
        *sampled_by_origin,
        "",
        "## Regenerate",
        "",
        "```bash",
        "devtools workspace claim-vs-evidence \\",
        "  --limit 5000 \\",
        "  --out-dir .agent/demos/claim-vs-evidence \\",
        "  --json",
        "devtools workspace demo-shelf",
        "```",
        "",
        "## Files",
        "",
        "- `claim-vs-evidence.report.json` — full machine-readable report.",
        "- `summary.json` — current demo-shelf claim/non-claim/proof/caveat summary.",
        "- `README.md` — this human-readable packet.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parsed = _parser().parse_args(argv)
    try:
        report = build_report(parsed)
    except ValueError as exc:
        print(f"claim-vs-evidence: {exc}", file=sys.stderr)
        return 2
    if parsed.json:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    elif parsed.out_dir is not None:
        print(f"wrote claim-vs-evidence artifacts to {parsed.out_dir}")
    else:
        totals = report["totals"]
        print(
            f"failed={totals['failed_outcomes']} acknowledged={totals['acknowledged']} "
            f"silent={totals['silent_proceed']} ambiguous={totals['ambiguous']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
