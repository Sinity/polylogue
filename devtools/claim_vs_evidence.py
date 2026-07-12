"""Build a focused claim-vs-evidence report from structured failures."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from polylogue.archive.actions.followup import classify_failed_followup_evidence
from polylogue.config import Config, get_config
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_WORDLESS_CONTINUATION_TEXT_CHAR_LIMIT = 40
_COUNT_KEYS = (
    "failed_outcomes",
    "acknowledged",
    "silent_proceed",
    "ambiguous",
    "ambiguous_wordless_continuation",
    "ambiguous_prose_no_marker",
)

# This is a methodology split, not a truth claim about any single tool result.
# Read/search tools often fail as part of ordinary path discovery; shell/build/
# edit tools are closer to consequential failures for a coding-agent workflow.
_BENIGN_RECOVERY_TOOLS = frozenset({"glob", "grep", "ls", "read"})
_CONSEQUENTIAL_TOOLS = frozenset(
    {
        "bash",
        "edit",
        "multi_edit",
        "notebook_edit",
        "patch",
        "run_command",
        "shell",
        "write",
    }
)
_CALIBRATION_LABELS = ("acknowledged", "silent_proceed", "ambiguous")
_CALIBRATION_SAMPLE_FILE = "ack-marker-calibration.sample.csv"
_CALIBRATION_LABELS_FILE = "ack-marker-calibration.labels.csv"
_PUBLIC_SUMMARY_FILE = "public-summary.json"
_PUBLIC_REPRODUCTION_FILE = "PUBLIC_REPRODUCTION.md"
_COLD_READER_GATE_FILE = "COLD_READER_GATE.md"
_DEFAULT_N_MIN = 30
_CALIBRATION_FIELDS = (
    "sample_id",
    "human_label",
    "classification",
    "classification_reason",
    "matched_marker",
    "origin",
    "model_name",
    "tool_name",
    "handler_class",
    "session_ref",
    "tool_result_message_ref",
    "next_message_ref",
    "next_text_preview",
    "next3_classification",
    "next3_matched_marker",
    "next3_text_preview",
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
    parser.add_argument(
        "--n-min",
        type=int,
        default=_DEFAULT_N_MIN,
        help="Minimum failures required before a split-cell rate is supported.",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=50,
        help="Deterministic stratified marker-calibration sample size to write.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=20260703,
        help="RNG seed for marker-calibration sample selection.",
    )
    parser.add_argument(
        "--calibration-labels",
        type=Path,
        default=None,
        help="Optional CSV of human labels. Defaults to ack-marker-calibration.labels.csv in --out-dir when present.",
    )
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


def _object_str_list(value: object) -> list[str]:
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return []


def _ranked(mapping: dict[str, dict[str, int]], *, n_min: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, counts in mapping.items():
        failed = counts["failed_outcomes"]
        silent = counts["silent_proceed"]
        classified = counts["acknowledged"] + silent
        supported = failed >= n_min
        classified_supported = classified >= n_min
        rows.append(
            {
                "name": name,
                **counts,
                "classified_outcomes": classified,
                "n_min": n_min,
                "coverage_status": "supported" if supported else "insufficient_n",
                "publication_status": "supported" if supported else "not_supported",
                "classified_coverage_status": "supported" if classified_supported else "insufficient_n",
                "classified_publication_status": "supported" if classified_supported else "not_supported",
                "silent_rate_lower_bound": (silent / failed) if supported else None,
                "silent_rate_among_classified": (silent / classified) if classified_supported else None,
            }
        )

    def sort_key(row: dict[str, object]) -> tuple[int, str]:
        failed = row["failed_outcomes"]
        failed_count = failed if isinstance(failed, int) else int(str(failed))
        return (-failed_count, str(row["name"]))

    return sorted(rows, key=sort_key)


def _empty_counts() -> dict[str, int]:
    return dict.fromkeys(_COUNT_KEYS, 0)


def _empty_window_counts() -> dict[str, int]:
    return {
        "failed_outcomes": 0,
        "acknowledged": 0,
        "silent_proceed": 0,
        "ambiguous": 0,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


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


def _handler_class(tool_name: str) -> str:
    normalized = tool_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _BENIGN_RECOVERY_TOOLS:
        return "benign_recovery"
    if normalized in _CONSEQUENTIAL_TOOLS:
        return "consequential"
    return "other"


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
                        rm.position AS result_position,
                        (
                            SELECT nm.message_id
                            FROM messages AS nm
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
                    JOIN messages AS rm ON rm.message_id = w.tool_result_message_id
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
                    p.result_position,
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


def _assistant_window_details(
    conn: Connection,
    rows: list[dict[str, object]],
    *,
    window_size: int = 3,
    chunk_size: int = 250,
) -> dict[int, dict[str, object]]:
    details: dict[int, dict[str, Any]] = {}
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        placeholders = ",".join("(?, ?, ?)" for _ in chunk)
        params: list[object] = []
        for index, row in enumerate(chunk, start=start):
            params.extend([index, row["session_id"], row["result_position"]])
        result_rows = _rows(
            conn,
            f"""
            WITH wanted(sort_index, session_id, result_position) AS (
                VALUES {placeholders}
            ),
            bounded AS (
                SELECT
                    w.*,
                    (
                        SELECT MIN(next_user.position)
                        FROM messages AS next_user
                        WHERE next_user.session_id = w.session_id
                          AND next_user.role = 'user'
                          AND next_user.position > w.result_position
                    ) AS next_user_position
                FROM wanted AS w
            ),
            assistant_window AS (
                SELECT
                    b.sort_index,
                    m.message_id,
                    m.position,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.sort_index
                        ORDER BY m.position
                    ) AS assistant_rank
                FROM bounded AS b
                JOIN messages AS m
                  ON m.session_id = b.session_id
                 AND m.role = 'assistant'
                 AND m.position > b.result_position
                 AND (
                    b.next_user_position IS NULL
                    OR m.position < b.next_user_position
                 )
            ),
            top_window AS (
                SELECT *
                FROM assistant_window
                WHERE assistant_rank <= ?
            )
            SELECT
                w.sort_index,
                w.message_id,
                w.assistant_rank,
                b.position AS block_position,
                b.block_type,
                COALESCE(b.text, '') AS text
            FROM top_window AS w
            LEFT JOIN blocks AS b ON b.message_id = w.message_id
            ORDER BY w.sort_index, w.assistant_rank, b.position
            """,
            [*params, window_size],
        )
        for result_row in result_rows:
            sort_index = _object_int(result_row["sort_index"])
            detail = details.setdefault(
                sort_index,
                {
                    "message_ids": [],
                    "text_parts": [],
                },
            )
            message_ids = detail["message_ids"]
            text_parts = detail["text_parts"]
            assert isinstance(message_ids, list)
            assert isinstance(text_parts, list)
            message_id = str(result_row["message_id"])
            if message_id not in message_ids:
                message_ids.append(message_id)
            if str(result_row["block_type"]) == "text":
                text_parts.append(str(result_row["text"] or ""))
    return {
        index: {
            "message_ids": detail["message_ids"],
            "text": "\n".join(str(part) for part in detail["text_parts"])[:2400],
        }
        for index, detail in details.items()
    }


def _structured_failure_rows(conn: Connection, *, limit: int, origin: str | None = None) -> list[dict[str, object]]:
    candidate_limit = max(limit * 2, limit + 100)
    rows = _paired_failure_rows(conn, _failure_outcome_rows(conn, limit=candidate_limit, origin=origin))[:limit]
    details = _next_message_details(conn, (row.get("next_message_id") for row in rows))
    window_details = _assistant_window_details(conn, rows)
    for index, row in enumerate(rows):
        detail = details.get(str(row.get("next_message_id") or ""), {})
        row["next_text"] = detail.get("next_text", "")
        row["next_has_tool_use"] = detail.get("next_has_tool_use", 0)
        row["next_pre_tool_text_chars"] = detail.get("next_pre_tool_text_chars", 0)
        window_detail = window_details.get(index, {})
        row["next3_message_ids"] = window_detail.get("message_ids", [])
        row["next3_text"] = window_detail.get("text", "")
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


def _calibration_sort_key(sample: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(sample["classification"]),
        str(sample["session_ref"]),
        str(sample["tool_result_message_ref"]),
    )


def _calibration_sample(
    samples: list[dict[str, object]],
    *,
    size: int,
    seed: int,
) -> list[dict[str, object]]:
    if size <= 0:
        return []
    by_class: dict[str, list[dict[str, object]]] = {label: [] for label in _CALIBRATION_LABELS}
    for sample in samples:
        classification = str(sample["classification"])
        if classification in by_class:
            by_class[classification].append(sample)
    rng = random.Random(seed)
    selected: list[dict[str, object]] = []
    target_per_class = max(1, size // len(_CALIBRATION_LABELS))
    for label in _CALIBRATION_LABELS:
        bucket = sorted(by_class[label], key=_calibration_sort_key)
        take = min(target_per_class, len(bucket))
        if take:
            selected.extend(rng.sample(bucket, take))
    if len(selected) < size:
        selected_ids = {
            (
                str(sample["session_ref"]),
                str(sample["tool_result_message_ref"]),
            )
            for sample in selected
        }
        remainder = [
            sample
            for sample in sorted(samples, key=_calibration_sort_key)
            if (
                str(sample["session_ref"]),
                str(sample["tool_result_message_ref"]),
            )
            not in selected_ids
        ]
        selected.extend(rng.sample(remainder, min(size - len(selected), len(remainder))))
    return sorted(selected[:size], key=_calibration_sort_key)


def _calibration_row(sample: Mapping[str, object], index: int, *, human_label: str = "") -> dict[str, object]:
    row = {field: sample.get(field, "") for field in _CALIBRATION_FIELDS}
    row["sample_id"] = f"cal-{index:03d}"
    row["human_label"] = human_label
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_CALIBRATION_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_calibration_labels(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {str(key): str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle)
            if row.get("human_label")
        ]


def _calibration_metrics(label_rows: list[dict[str, str]], *, labels_path: Path | None) -> dict[str, object]:
    confusion: dict[str, dict[str, int]] = {
        human: dict.fromkeys(_CALIBRATION_LABELS, 0) for human in _CALIBRATION_LABELS
    }
    invalid_rows = 0
    for row in label_rows:
        human = row.get("human_label", "").strip()
        predicted = row.get("classification", "").strip()
        if human not in confusion or predicted not in _CALIBRATION_LABELS:
            invalid_rows += 1
            continue
        confusion[human][predicted] += 1
    acknowledged_tp = confusion["acknowledged"]["acknowledged"]
    predicted_acknowledged = sum(confusion[human]["acknowledged"] for human in _CALIBRATION_LABELS)
    human_acknowledged = sum(confusion["acknowledged"].values())
    usable_rows = sum(sum(row.values()) for row in confusion.values())
    return {
        "labels_path": str(labels_path) if labels_path is not None else None,
        "labeled_rows": usable_rows,
        "invalid_rows": invalid_rows,
        "labels": list(_CALIBRATION_LABELS),
        "confusion_matrix": confusion,
        "ack_marker_precision": _rate(acknowledged_tp, predicted_acknowledged),
        "ack_marker_recall": _rate(acknowledged_tp, human_acknowledged),
        "ack_marker_true_positive": acknowledged_tp,
        "ack_marker_predicted_positive": predicted_acknowledged,
        "ack_marker_actual_positive": human_acknowledged,
    }


def _calibration_labels_path(args: argparse.Namespace) -> Path | None:
    calibration_labels = args.calibration_labels
    if isinstance(calibration_labels, Path):
        return calibration_labels
    out_dir = args.out_dir
    if not isinstance(out_dir, Path):
        return None
    candidate = out_dir / _CALIBRATION_LABELS_FILE
    return candidate if candidate.exists() else None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.sample_limit < 1:
        raise ValueError("--sample-limit must be positive")
    if args.n_min < 1:
        raise ValueError("--n-min must be positive")
    if args.calibration_size < 0:
        raise ValueError("--calibration-size must be non-negative")
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
    window3_totals = _empty_window_counts()
    by_tool: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    by_origin: dict[str, dict[str, int]] = {}
    by_handler_class: dict[str, dict[str, int]] = {}
    samples_by_classification: dict[str, list[dict[str, object]]] = {
        "acknowledged": [],
        "silent_proceed": [],
        "ambiguous": [],
    }
    samples_by_origin_classification: dict[str, dict[str, list[dict[str, object]]]] = {}
    calibration_candidates: list[dict[str, object]] = []
    for row in rows:
        classification_evidence = classify_failed_followup_evidence(
            str(row["next_text"]) if row["next_text"] is not None else None
        )
        classification = str(classification_evidence["classification"])
        classification_reason = _refine_classification_reason(classification_evidence, row)
        next3_message_ids = _object_str_list(row["next3_message_ids"])
        next3_text = str(row["next3_text"] or "")
        window3_evidence = classify_failed_followup_evidence(next3_text if next3_message_ids else None)
        window3_classification = str(window3_evidence["classification"])
        tool = str(row["tool_name"] or "unknown")
        handler_class = _handler_class(tool)
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
            "handler_class": handler_class,
            "model_name": model,
            "origin": origin,
            "exit_code": row["exit_code"],
            "is_error": row["is_error"],
            "tool_command_preview": str(row["tool_command"] or "")[:160],
            "next_text_preview": next_text[:500],
            "next_has_tool_use": bool(_object_int(row["next_has_tool_use"])),
            "next_pre_tool_text_chars": _object_int(row["next_pre_tool_text_chars"]),
            "next3_message_refs": [f"message:{message_id}" for message_id in next3_message_ids],
            "next3_classification": window3_classification,
            "next3_classification_reason": window3_evidence["reason"],
            "next3_matched_marker": window3_evidence["matched_marker"],
            "next3_text_preview": next3_text[:500],
        }
        totals["failed_outcomes"] += 1
        totals[classification] += 1
        window3_totals["failed_outcomes"] += 1
        window3_totals[window3_classification] += 1
        ambiguous_counter_key = _ambiguous_counter_key(classification_reason)
        if ambiguous_counter_key is not None:
            totals[ambiguous_counter_key] += 1
        by_tool.setdefault(tool, _empty_counts())
        by_model.setdefault(model, _empty_counts())
        by_origin.setdefault(origin, _empty_counts())
        by_handler_class.setdefault(handler_class, _empty_counts())
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
        by_handler_class[handler_class]["failed_outcomes"] += 1
        by_handler_class[handler_class][classification] += 1
        if ambiguous_counter_key is not None:
            by_handler_class[handler_class][ambiguous_counter_key] += 1
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
        calibration_candidates.append(sample)
    totals["classified_outcomes"] = totals["acknowledged"] + totals["silent_proceed"]
    window3_totals["classified_outcomes"] = window3_totals["acknowledged"] + window3_totals["silent_proceed"]
    silent = totals["silent_proceed"]
    failed = totals["failed_outcomes"]
    classified = totals["classified_outcomes"]
    aggregate_supported = failed >= args.n_min
    window3_silent = window3_totals["silent_proceed"]
    window3_classified = window3_totals["classified_outcomes"]
    calibration_sample = _calibration_sample(
        calibration_candidates,
        size=args.calibration_size,
        seed=args.calibration_seed,
    )
    calibration_labels_path = _calibration_labels_path(args)
    calibration_label_rows = (
        _read_calibration_labels(calibration_labels_path)
        if calibration_labels_path is not None and calibration_labels_path.exists()
        else []
    )
    calibration = {
        "sample_size_requested": args.calibration_size,
        "sample_size": len(calibration_sample),
        "sample_seed": args.calibration_seed,
        "sample_file": _CALIBRATION_SAMPLE_FILE if args.out_dir is not None else None,
        "labels_file": _CALIBRATION_LABELS_FILE if args.out_dir is not None else None,
        "metrics": _calibration_metrics(calibration_label_rows, labels_path=calibration_labels_path),
    }
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
            "sensitivity_scope": "next 3 assistant messages after the failed result, stopping before the next user message",
            "n_min": args.n_min,
            "thin_cell_policy": (
                "Split cells below n_min are retained for coverage accounting but publish no rates: "
                "coverage_status=insufficient_n and publication_status=not_supported; "
                "classified-denominator rates independently require classified_outcomes >= n_min."
            ),
            "total_by_origin": total_by_origin,
            "sampled_by_origin": sampled_by_origin,
        },
        "definition": (
            "Structured failures are normalized tool-result outcomes with is_error=1 or non-zero exit_code. "
            "The immediately following assistant message is classified only for explicit failure "
            "acknowledgment markers; this is not an LLM judgment or prose-mined outcome."
        ),
        "totals": totals,
        "window3_totals": window3_totals,
        "rates": {
            "coverage_status": "supported" if aggregate_supported else "insufficient_n",
            "publication_status": "supported" if aggregate_supported else "not_supported",
            "classified_coverage_status": "supported" if classified >= args.n_min else "insufficient_n",
            "classified_publication_status": "supported" if classified >= args.n_min else "not_supported",
            "window3_classified_coverage_status": "supported" if window3_classified >= args.n_min else "insufficient_n",
            "window3_classified_publication_status": "supported"
            if window3_classified >= args.n_min
            else "not_supported",
            "n_min": args.n_min,
            "silent_rate_lower_bound": (silent / failed) if aggregate_supported else None,
            "silent_rate_among_classified": (silent / classified) if classified >= args.n_min else None,
            "window3_silent_rate_lower_bound": (window3_silent / failed) if aggregate_supported else None,
            "window3_silent_rate_among_classified": (
                (window3_silent / window3_classified) if window3_classified >= args.n_min else None
            ),
            "ack_later_within_3": max(0, window3_totals["acknowledged"] - totals["acknowledged"]),
        },
        "by_tool": _ranked(by_tool, n_min=args.n_min),
        "by_model": _ranked(by_model, n_min=args.n_min),
        "by_origin": _ranked(by_origin, n_min=args.n_min),
        "by_handler_class": _ranked(by_handler_class, n_min=args.n_min),
        "handler_class_definition": {
            "benign_recovery": sorted(_BENIGN_RECOVERY_TOOLS),
            "consequential": sorted(_CONSEQUENTIAL_TOOLS),
            "other": "Any tool name outside the explicit benign/consequential methodology sets.",
        },
        "calibration": calibration,
        "calibration_sample": [
            _calibration_row(sample, index) for index, sample in enumerate(calibration_sample, start=1)
        ],
        "samples_by_classification": samples_by_classification,
        "samples_by_origin_classification": samples_by_origin_classification,
    }
    if args.out_dir is not None:
        _write_artifacts(args.out_dir, report)
    return report


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format_rate_percent(value: int | float | str | None) -> str:
    if value is None:
        return "not enough labels"
    return f"{float(value):.1%}"


def _public_summary(report: dict[str, Any]) -> dict[str, Any]:
    totals = report["totals"]
    frame = report["sample_frame"]
    rates = report["rates"]
    calibration_metrics = report["calibration"]["metrics"]
    return {
        "artifact": "claim-vs-evidence-public-summary",
        "generated_at": report["captured_at"],
        "archive_root": report["archive_root"],
        "index_schema_version": report["index_schema_version"],
        "claim": (
            "Polylogue can ground a failure-follow-up finding in normalized tool-result outcomes, "
            "state the bounded sample frame, and publish aggregate rates without exposing raw private transcripts."
        ),
        "non_claim": (
            "The live aggregate is not reproducible without the private archive; the deterministic demo archive "
            "reproduces the method and artifact shape, not the private corpus rates."
        ),
        "proofs": [
            {
                "name": "structured_failure_frame",
                "total_structured_failures": frame["total_structured_failures"],
                "inspected_structured_failures": frame["inspected_structured_failures"],
                "unpaired_structured_failures": frame["unpaired_structured_failures"],
                "selection_strategy": frame["selection_strategy"],
                "selection_order": frame["selection_order"],
                "n_min": frame["n_min"],
                "thin_cell_policy": frame["thin_cell_policy"],
            },
            {
                "name": "next_turn_classification",
                "acknowledged": totals["acknowledged"],
                "silent_proceed": totals["silent_proceed"],
                "ambiguous": totals["ambiguous"],
                "silent_rate_lower_bound": rates["silent_rate_lower_bound"],
                "silent_rate_among_classified": rates["silent_rate_among_classified"],
            },
            {
                "name": "sensitivity_and_calibration",
                "ack_later_within_3": rates["ack_later_within_3"],
                "window3_silent_rate_lower_bound": rates["window3_silent_rate_lower_bound"],
                "calibration_labeled_rows": calibration_metrics["labeled_rows"],
                "ack_marker_precision": calibration_metrics["ack_marker_precision"],
                "ack_marker_recall": calibration_metrics["ack_marker_recall"],
            },
        ],
        "caveats": [
            "Private live-archive counts are aggregate-only in this public summary.",
            "Deterministic demo reproduction validates the method and renderer, not the private rate estimates.",
            "The classifier is an explicit marker detector; ambiguous rows remain in the denominator.",
            "The report is bounded by --limit unless the limit exceeds the full structured-failure frame.",
            "Split cells below n_min are coverage-only and explicitly not supported for rate publication.",
        ],
        "reproduction": {
            "demo_archive_root": "/realm/tmp/polylogue-claim-vs-evidence-demo",
            "commands": [
                "export POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/polylogue-claim-vs-evidence-demo",
                'polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json',
                'polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json',
                ("polylogue --plain --format json actions where is_error:true \\| group by followup_class \\| count"),
                "polylogue --plain --format json actions where followup_class:silent_proceed",
                (
                    "devtools workspace claim-vs-evidence "
                    '--archive-root "$POLYLOGUE_ARCHIVE_ROOT" '
                    "--limit 5000 --out-dir /realm/tmp/polylogue-claim-vs-evidence-repro --json"
                ),
            ],
            "shared_queries": [
                "actions where is_error:true | group by followup_class | count",
                "actions where followup_class:silent_proceed",
            ],
        },
    }


def _write_public_reproduction(path: Path, report: dict[str, Any]) -> None:
    summary = _public_summary(report)
    proof_by_name = {str(item["name"]): item for item in summary["proofs"]}
    frame = proof_by_name["structured_failure_frame"]
    classification = proof_by_name["next_turn_classification"]
    sensitivity = proof_by_name["sensitivity_and_calibration"]
    commands = "\n".join(summary["reproduction"]["commands"])
    path.write_text(
        "\n".join(
            [
                "# Claim-vs-Evidence Public Reproduction",
                "",
                "This packet is the public-safe wrapper for the live claim-vs-evidence demo.",
                "It contains aggregate live-archive findings and a private-data-free reproduction",
                "path over Polylogue's deterministic demo archive.",
                "",
                "## What A Reader Can Claim",
                "",
                str(summary["claim"]),
                "",
                "## What A Reader Cannot Claim",
                "",
                str(summary["non_claim"]),
                "",
                "## Live Aggregate Evidence",
                "",
                f"- archive root: `{report['archive_root']}`",
                f"- index schema: v{report['index_schema_version']}",
                f"- total structured failures: {int(frame['total_structured_failures']):,}",
                f"- inspected structured failures: {int(frame['inspected_structured_failures']):,}",
                f"- unpaired structured failures: {int(frame['unpaired_structured_failures']):,}",
                f"- acknowledged next turn: {int(classification['acknowledged']):,}",
                f"- silent-proceed next turn: {int(classification['silent_proceed']):,}",
                f"- ambiguous next turn: {int(classification['ambiguous']):,}",
                f"- silent lower bound: {_format_rate_percent(classification['silent_rate_lower_bound'])}",
                f"- next-3 silent lower bound: {_format_rate_percent(sensitivity['window3_silent_rate_lower_bound'])}",
                f"- calibration labeled rows: {int(sensitivity['calibration_labeled_rows']):,}",
                f"- acknowledged-marker precision: {_format_rate_percent(sensitivity['ack_marker_precision'])}",
                f"- acknowledged-marker recall: {_format_rate_percent(sensitivity['ack_marker_recall'])}",
                "",
                "## Shared Query Form",
                "",
                "The core next-turn counts are ordinary action-unit queries, not report-private SQL:",
                "",
                "```text",
                *[str(query) for query in summary["reproduction"]["shared_queries"]],
                "```",
                "",
                "## Reproduce The Method Without Private Data",
                "",
                "```bash",
                commands,
                "```",
                "",
                "The reproduction output should contain the same artifact family:",
                "`claim-vs-evidence.report.json`, `summary.json`, `README.md`,",
                f"`{_PUBLIC_SUMMARY_FILE}`, and this public reproduction contract.",
                "Counts will differ because the deterministic demo archive is synthetic.",
                "",
                "## Caveats",
                "",
                *[f"- {item}" for item in summary["caveats"]],
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_cold_reader_gate(path: Path, report: dict[str, Any]) -> None:
    summary = _public_summary(report)
    path.write_text(
        "\n".join(
            [
                "# Cold-Reader Gate",
                "",
                "Give a fresh reader only this directory and ask:",
                "",
                "```text",
                "Using only the files in this directory, state what the claim-vs-evidence",
                "artifact proves, what it does not prove, what sample frame it used,",
                "how to reproduce the method without private data, and the most important",
                "caveats before quoting any rate.",
                "```",
                "",
                "## Expected Passing Answer",
                "",
                "- Names structured tool-result outcomes as the failure evidence anchor.",
                "- States that the live rate is aggregate private-archive evidence, not a seeded-corpus rate.",
                "- Includes archive root, index schema, total failures, inspected failures, and unpaired failures.",
                "- Reports next-turn silent lower bound, next-3 sensitivity, and calibration precision/recall.",
                "- Explains that the deterministic demo archive reproduces the method and artifact shape only.",
                "- Mentions ambiguous rows remain in the denominator and the classifier is marker-based.",
                "",
                "## Current Gate Evidence",
                "",
                f"- public summary: `{_PUBLIC_SUMMARY_FILE}`",
                f"- public reproduction: `{_PUBLIC_REPRODUCTION_FILE}`",
                f"- aggregate live archive root: `{summary['archive_root']}`",
                f"- aggregate index schema: v{summary['index_schema_version']}",
                "- status: ready for an external cold read; no private transcript previews are required.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_artifacts(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "claim-vs-evidence.report.json", report)
    _write_csv(out_dir / _CALIBRATION_SAMPLE_FILE, report["calibration_sample"])
    _write_json(out_dir / _PUBLIC_SUMMARY_FILE, _public_summary(report))
    _write_public_reproduction(out_dir / _PUBLIC_REPRODUCTION_FILE, report)
    _write_cold_reader_gate(out_dir / _COLD_READER_GATE_FILE, report)
    totals = report["totals"]
    window3_totals = report["window3_totals"]
    rates = report["rates"]
    calibration = report["calibration"]
    calibration_metrics = calibration["metrics"]
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
            "acknowledged_within_3": window3_totals["acknowledged"],
            "silent_proceed_within_3": window3_totals["silent_proceed"],
            "ambiguous_within_3": window3_totals["ambiguous"],
            "ack_later_within_3": rates["ack_later_within_3"],
            "ambiguous_wordless_continuation": totals["ambiguous_wordless_continuation"],
            "ambiguous_prose_no_marker": totals["ambiguous_prose_no_marker"],
            "silent_rate_lower_bound": rates["silent_rate_lower_bound"],
            "window3_silent_rate_lower_bound": rates["window3_silent_rate_lower_bound"],
            "by_handler_class": report["by_handler_class"],
            "handler_class_definition": report["handler_class_definition"],
            "limit": report["limit"],
            "time_window": report["sample_frame"]["time_window"],
            "sensitivity_scope": report["sample_frame"]["sensitivity_scope"],
            "sampled_by_origin": report["sample_frame"]["sampled_by_origin"],
            "calibration": {
                "sample_size": calibration["sample_size"],
                "sample_seed": calibration["sample_seed"],
                "labeled_rows": calibration_metrics["labeled_rows"],
                "ack_marker_precision": calibration_metrics["ack_marker_precision"],
                "ack_marker_recall": calibration_metrics["ack_marker_recall"],
            },
        },
        "caveats": [
            "The report is bounded by --limit for fast active-archive regeneration.",
            "The headline classification inspects only the next assistant message; the next-3 window is a sensitivity row.",
            "Marker calibration is based on the committed label CSV when present; unlabeled sample rows do not count.",
            "Structured failure truth comes from normalized action result is_error/exit_code fields, not assistant prose.",
            "Failed tool results without a paired tool-use row are reported as unpaired coverage gaps, not classified rows.",
        ],
        "source_files": [
            "claim-vs-evidence.report.json",
            _PUBLIC_SUMMARY_FILE,
            _PUBLIC_REPRODUCTION_FILE,
            _COLD_READER_GATE_FILE,
            _CALIBRATION_SAMPLE_FILE,
            _CALIBRATION_LABELS_FILE,
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    _write_readme(out_dir / "README.md", report)


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    totals = report["totals"]
    window3_totals = report["window3_totals"]
    rates = report["rates"]
    frame = report["sample_frame"]
    calibration = report["calibration"]
    calibration_metrics = calibration["metrics"]
    precision = calibration_metrics["ack_marker_precision"]
    recall = calibration_metrics["ack_marker_recall"]
    sampled_by_origin = [
        (
            f"- {row['origin']}: inspected {int(row['inspected_structured_failures']):,} / "
            f"{int(row['total_structured_failures']):,} structured failures "
            f"(requested {int(row['requested_limit']):,})"
        )
        for row in frame["sampled_by_origin"]
    ]
    handler_class_rows = [
        (
            f"- {row['name']}: failed {int(row['failed_outcomes']):,}; "
            f"silent {int(row['silent_proceed']):,}; "
            f"ambiguous {int(row['ambiguous']):,}; "
            + (
                f"silent lower bound {float(row['silent_rate_lower_bound']):.1%}"
                if row["silent_rate_lower_bound"] is not None
                else f"not supported (n < {int(row['n_min'])})"
            )
        )
        for row in report["by_handler_class"]
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
        f"- silent lower bound: {_format_rate_percent(rates['silent_rate_lower_bound'])}",
        f"- silent among classified: {_format_rate_percent(rates['silent_rate_among_classified'])}",
        f"- acknowledged within next 3 assistant turns: {window3_totals['acknowledged']:,}",
        f"- acknowledgments appearing only after the next turn: {rates['ack_later_within_3']:,}",
        f"- silent lower bound after next-3 sensitivity: {_format_rate_percent(rates['window3_silent_rate_lower_bound'])}",
        f"- configured limit: {report['limit']:,}",
        f"- split-cell minimum n: {frame['n_min']:,} (below this, rates are not supported)",
        f"- selection order: {frame['selection_order']}",
        f"- selection strategy: {frame['selection_strategy']}",
        "",
        "### Handler-Class Split",
        "",
        "The headline should not mix ordinary read/search recovery with more consequential",
        "shell/build/edit failures without saying so. Handler classes are explicit and",
        "methodological: `benign_recovery` covers read/search/path-discovery tools,",
        "`consequential` covers shell/build/edit/write-class tools, and `other` is not",
        "folded into either claim.",
        "",
        *handler_class_rows,
        "",
        "### Inspected vs Total by Origin",
        "",
        *sampled_by_origin,
        "",
        "### Marker Calibration",
        "",
        "The classifier is an explicit marker detector, not an LLM judgment. This",
        "calibration sample is deterministic and stratified across acknowledged,",
        "silent-proceed, and ambiguous predicted classes. Human labels, when present,",
        "live in `ack-marker-calibration.labels.csv` so regeneration does not overwrite",
        "manual judgment.",
        "",
        f"- calibration sample size: {int(calibration['sample_size']):,}",
        f"- calibration seed: {int(calibration['sample_seed'])}",
        f"- labeled rows: {int(calibration_metrics['labeled_rows']):,}",
        (
            f"- acknowledged-marker precision: {float(precision):.1%}"
            if precision is not None
            else "- acknowledged-marker precision: not enough labels"
        ),
        (
            f"- acknowledged-marker recall: {float(recall):.1%}"
            if recall is not None
            else "- acknowledged-marker recall: not enough labels"
        ),
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
        f"- `{_PUBLIC_SUMMARY_FILE}` — aggregate-only public-safe summary.",
        f"- `{_PUBLIC_REPRODUCTION_FILE}` — seeded private-data-free reproduction instructions.",
        f"- `{_COLD_READER_GATE_FILE}` — cold-reader prompt and passing-answer checklist.",
        f"- `{_CALIBRATION_SAMPLE_FILE}` — deterministic sample for marker calibration.",
        f"- `{_CALIBRATION_LABELS_FILE}` — optional human labels consumed on regeneration.",
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
