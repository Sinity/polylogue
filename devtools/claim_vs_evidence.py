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


def _structured_failure_rows(conn: Connection, *, limit: int) -> list[dict[str, object]]:
    return _rows(
        conn,
        """
        WITH failed AS (
            SELECT
                r.session_id,
                u.message_id,
                u.tool_name,
                u.tool_command,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                m.position,
                m.model_name AS tool_message_model
            FROM blocks AS r INDEXED BY idx_blocks_type
            JOIN blocks AS u INDEXED BY idx_blocks_tool_id
              ON u.tool_id = r.tool_id
             AND u.session_id = r.session_id
             AND u.block_type = 'tool_use'
            JOIN messages AS m ON m.message_id = u.message_id
            WHERE r.block_type = 'tool_result'
              AND (
                  COALESCE(r.tool_result_is_error, 0) = 1
                  OR (r.tool_result_exit_code IS NOT NULL AND r.tool_result_exit_code != 0)
              )
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
            n.tool_name,
            n.tool_command,
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
            n.tool_name,
            n.tool_command,
            n.is_error,
            n.exit_code,
            model_name,
            n.next_message_id
        """,
        (limit,),
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")
    config = _config_with_archive_root(get_config(), args.archive_root)
    index_db = config.db_path
    conn = open_readonly_connection(index_db)
    try:
        rows = _structured_failure_rows(conn, limit=args.limit)
        schema_version = _user_version(conn)
    finally:
        conn.close()

    totals = {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0}
    by_tool: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    samples_by_classification: dict[str, list[dict[str, object]]] = {
        "acknowledged": [],
        "silent_proceed": [],
        "ambiguous": [],
    }
    for row in rows:
        classification = _classify_failed_followup(str(row["next_text"]) if row["next_text"] is not None else None)
        tool = str(row["tool_name"] or "unknown")
        model = str(row["model_name"] or "unknown")
        sample = {
            "classification": classification,
            "session_ref": f"session:{row['session_id']}",
            "tool_message_ref": f"message:{row['message_id']}",
            "next_message_ref": f"message:{row['next_message_id']}" if row["next_message_id"] else None,
            "tool_name": tool,
            "model_name": model,
            "exit_code": row["exit_code"],
            "is_error": row["is_error"],
            "tool_command_preview": str(row["tool_command"] or "")[:160],
        }
        totals["failed_outcomes"] += 1
        totals[classification] += 1
        by_tool.setdefault(tool, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_model.setdefault(model, {"failed_outcomes": 0, "acknowledged": 0, "silent_proceed": 0, "ambiguous": 0})
        by_tool[tool]["failed_outcomes"] += 1
        by_tool[tool][classification] += 1
        by_model[model]["failed_outcomes"] += 1
        by_model[model][classification] += 1
        bucket = samples_by_classification[classification]
        if len(bucket) < args.sample_limit:
            bucket.append(sample)
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
        "definition": (
            "Structured failures are tool_result rows with is_error=1 or non-zero exit_code. "
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
        "samples_by_classification": samples_by_classification,
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
            "acknowledged": totals["acknowledged"],
            "silent_proceed": totals["silent_proceed"],
            "ambiguous": totals["ambiguous"],
            "silent_rate_lower_bound": rates["silent_rate_lower_bound"],
            "limit": report["limit"],
        },
        "caveats": [
            "The report is bounded by --limit for fast active-archive regeneration.",
            "Classification inspects only the next assistant message for explicit acknowledgment markers.",
            "Structured failure truth comes from tool_result is_error/exit_code fields, not assistant prose.",
        ],
        "source_files": ["claim-vs-evidence.report.json"],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_readme(out_dir / "README.md", report)


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    totals = report["totals"]
    rates = report["rates"]
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
        "predicate is `is_error=1` or a non-zero `exit_code` on `tool_result` rows.",
        "",
        "## Current Bounded Result",
        "",
        f"- failed structured outcomes inspected: {totals['failed_outcomes']:,}",
        f"- acknowledged: {totals['acknowledged']:,}",
        f"- silent-proceed: {totals['silent_proceed']:,}",
        f"- ambiguous: {totals['ambiguous']:,}",
        f"- silent lower bound: {rates['silent_rate_lower_bound']:.1%}",
        f"- silent among classified: {rates['silent_rate_among_classified']:.1%}",
        f"- configured limit: {report['limit']:,}",
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
