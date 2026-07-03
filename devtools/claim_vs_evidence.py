"""Build a focused claim-vs-evidence report from structured failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from polylogue.config import Config, get_config
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from scripts.agent_forensics import _classify_failed_followup


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


def _structured_failure_rows(conn: Connection, *, limit: int, origin: str | None = None) -> list[dict[str, object]]:
    origin_predicate = "AND s.origin = ?" if origin is not None else ""
    params: tuple[object, ...] = (origin, limit) if origin is not None else (limit,)
    return _rows(
        conn,
        f"""
        WITH failed AS (
            SELECT DISTINCT
                r.session_id,
                u.message_id,
                r.message_id AS tool_result_message_id,
                u.tool_name,
                u.tool_command,
                s.origin,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                COALESCE(rm.position, m.position) AS position,
                m.model_name AS tool_message_model
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN blocks AS u INDEXED BY idx_blocks_tool_id
              ON u.tool_id = r.tool_id
             AND u.session_id = r.session_id
             AND u.block_type = 'tool_use'
            JOIN sessions AS s ON s.session_id = r.session_id
            JOIN messages AS m ON m.message_id = u.message_id
            JOIN messages AS rm ON rm.message_id = r.message_id
            WHERE r.block_type = 'tool_result'
              {origin_predicate}
              AND (
                  COALESCE(r.tool_result_is_error, 0) = 1
                  OR (r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0)
              )
            ORDER BY r.session_id, COALESCE(rm.position, m.position), r.message_id, r.tool_id
            LIMIT ?
        ),
        next_message AS (
            SELECT
                f.*,
                (
                    SELECT nm.message_id
                    FROM messages AS nm
                    WHERE nm.session_id = f.session_id
                      AND nm.role = 'assistant'
                      AND nm.position > f.position
                    ORDER BY nm.position
                    LIMIT 1
                ) AS next_message_id
            FROM failed AS f
        )
        SELECT
            n.session_id,
            n.message_id,
            n.tool_result_message_id,
            n.tool_name,
            n.tool_command,
            n.origin,
            n.is_error,
            n.exit_code,
            COALESCE(nm.model_name, n.tool_message_model, '') AS model_name,
            n.next_message_id,
            substr(group_concat(COALESCE(b.text, ''), '\n'), 1, 1200) AS next_text
        FROM next_message AS n
        LEFT JOIN messages AS nm ON nm.message_id = n.next_message_id
        LEFT JOIN blocks AS b ON b.message_id = n.next_message_id
        GROUP BY
            n.session_id,
            n.message_id,
            n.tool_result_message_id,
            n.tool_name,
            n.tool_command,
            n.origin,
            n.is_error,
            n.exit_code,
            model_name,
            n.next_message_id
        """,
        params,
    )


def _structured_failure_count(conn: Connection) -> int:
    return _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM (
            SELECT r.session_id, r.message_id, r.tool_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN blocks AS u INDEXED BY idx_blocks_tool_id
              ON u.tool_id = r.tool_id
             AND u.session_id = r.session_id
             AND u.block_type = 'tool_use'
            WHERE r.block_type = 'tool_result'
              AND (
                  COALESCE(r.tool_result_is_error, 0) = 1
                  OR (r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0)
              )
            GROUP BY r.session_id, r.message_id, r.tool_id
        )
        """,
    )


def _unpaired_structured_failure_count(conn: Connection) -> int:
    return _scalar_int(
        conn,
        """
        SELECT COUNT(*)
        FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
        WHERE r.block_type = 'tool_result'
          AND (
              COALESCE(r.tool_result_is_error, 0) = 1
              OR (r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0)
          )
          AND NOT EXISTS (
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
        SELECT origin, COUNT(*) AS failed_outcomes
        FROM (
            SELECT s.origin, r.session_id, r.message_id, r.tool_id
            FROM blocks AS r INDEXED BY idx_blocks_tool_result_outcome
            JOIN blocks AS u INDEXED BY idx_blocks_tool_id
              ON u.tool_id = r.tool_id
             AND u.session_id = r.session_id
             AND u.block_type = 'tool_use'
            JOIN sessions AS s ON s.session_id = r.session_id
            WHERE r.block_type = 'tool_result'
              AND (
                  COALESCE(r.tool_result_is_error, 0) = 1
                  OR (r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0)
              )
            GROUP BY s.origin, r.session_id, r.message_id, r.tool_id
        )
        GROUP BY origin
        ORDER BY failed_outcomes DESC, origin
        """,
    )


def _origin_sample_limits(total_by_origin: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    origins = [(str(row["origin"]), int(row["failed_outcomes"])) for row in total_by_origin]
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
        total_structured_failures = _structured_failure_count(conn)
        unpaired_structured_failures = _unpaired_structured_failure_count(conn)
        total_by_origin = _structured_failure_origin_counts(conn)
        origin_limits = _origin_sample_limits(total_by_origin, args.limit)
        rows = []
        sampled_by_origin: list[dict[str, object]] = []
        for origin_limit in origin_limits:
            origin = str(origin_limit["origin"])
            requested_limit = int(origin_limit["requested_limit"])
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

    totals = {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0}
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
        classification = _classify_failed_followup(str(row["next_text"]) if row["next_text"] is not None else None)
        tool = str(row["tool_name"] or "unknown")
        model = str(row["model_name"] or "unknown")
        origin = str(row["origin"] or "unknown")
        next_text = str(row["next_text"] or "")
        sample = {
            "classification": classification,
            "session_ref": f"session:{row['session_id']}",
            "tool_message_ref": f"message:{row['message_id']}",
            "tool_result_message_ref": f"message:{row['tool_result_message_id']}",
            "next_message_ref": f"message:{row['next_message_id']}" if row["next_message_id"] else None,
            "tool_name": tool,
            "model_name": model,
            "origin": origin,
            "exit_code": row["exit_code"],
            "is_error": row["is_error"],
            "tool_command_preview": str(row["tool_command"] or "")[:160],
            "next_text_preview": next_text[:500],
        }
        totals["failed_outcomes"] += 1
        totals[classification] += 1
        by_tool.setdefault(tool, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_model.setdefault(model, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_origin.setdefault(origin, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_tool[tool]["failed_outcomes"] += 1
        by_tool[tool][classification] += 1
        by_model[model]["failed_outcomes"] += 1
        by_model[model][classification] += 1
        by_origin[origin]["failed_outcomes"] += 1
        by_origin[origin][classification] += 1
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
            "complete_failure_frame": len(rows) >= total_structured_failures,
            "selection_strategy": (
                "origin-stratified bounded sample; at least one row per origin when limit allows, "
                "then proportional fill by origin failure count"
            ),
            "selection_order": "origin, session_id, tool_result_message_position, tool_result_message_id, tool_id",
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
            "silent_rate_lower_bound": rates["silent_rate_lower_bound"],
            "limit": report["limit"],
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
        f"- total structured failures in frame: {frame['total_structured_failures']:,}",
        f"- unpaired structured failures outside classifiable frame: {frame['unpaired_structured_failures']:,}",
        f"- failed structured outcomes inspected: {totals['failed_outcomes']:,}",
        f"- complete failure frame: {frame['complete_failure_frame']}",
        f"- acknowledged: {totals['acknowledged']:,}",
        f"- silent-proceed: {totals['silent_proceed']:,}",
        f"- ambiguous: {totals['ambiguous']:,}",
        f"- silent lower bound: {rates['silent_rate_lower_bound']:.1%}",
        f"- silent among classified: {rates['silent_rate_among_classified']:.1%}",
        f"- configured limit: {report['limit']:,}",
        f"- selection order: {frame['selection_order']}",
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
